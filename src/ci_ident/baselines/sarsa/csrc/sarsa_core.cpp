// Tabular SARSA selection of CI instances (after Wang and Xiao, 2023).
// Python module: ci_ident.baselines.sarsa._sarsa_core
//
// MDP.  A state is the set of instances selected so far (interned by its
// sorted index list).  The actions in a state are the instances that are in
// conflict with none of the selected ones and fit the area budget:
//   * node conflict -- the instance shares a node with a selected instance;
//   * dependency conflict -- contracting every selected instance and the
//     candidate into single nodes creates a directed cycle in the DDG, i.e.
//     the candidate together with the selection cannot be scheduled;
//   * area -- the candidate's pattern is not yet implemented and its area
//     would take the total above the budget (area is charged once per
//     pattern).
// The reward of an action is the instance's whole-cycle saving; an episode
// ends when no action is left.  Every relation is monotone in the selection
// (a candidate excluded once stays excluded for the rest of the episode),
// so the action set is evaluated lazily: candidates are tested when they are
// drawn or scanned and excluded ones are marked for the episode.  This
// yields exactly the action sets of an eagerly filtered conflict graph whose
// dependency relation is taken over the whole selection.
//
// The dependency test runs on the full DDG (every node, admissible or not).
// It keeps, per DDG node u, the set Rch(u) of nodes reachable from u in the
// DDG with the selected instances contracted.  A candidate C closes a cycle
// iff some successor s outside C of a node of C has Rch(s) meeting C.  On
// selection of C, every node that reaches C (and C itself) gains
// C and the union of Rch over C.
//
// Learning.  SARSA(0) with an on-policy next action:
//   Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a') - Q(s,a) ]
// (target r at the end of the episode), epsilon-greedy behaviour with
// epsilon(ep) = max(eps_end, eps_start * (1 - ep / episodes)), Q initialised
// to 0, greedy ties broken by the lowest instance index (instances are passed
// in priority order).  After training, a greedy rollout (epsilon = 0) on the
// learned Q gives the selection.  Random draws use one std::mt19937_64
// seeded with `seed`, so a run is a function of its inputs and the number of
// episodes completed.
//
// Time.  `seconds` is a duration from the call on the monotonic clock (+inf:
// none, <= 0: already expired).  Training stops when fewer than two mean
// episode durations remain, leaving time for the rollout; any episode or the
// rollout is cut at the deadline itself (a cut rollout returns its prefix,
// which is itself a valid selection).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

namespace {

struct Deadline {
    bool bounded = false;
    Clock::time_point at{};
    explicit Deadline(double seconds) {
        if (std::isfinite(seconds)) {
            bounded = true;
            double s = seconds > 0.0 ? seconds : 0.0;
            at = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(s));
        }
    }
    bool expired() const { return bounded && Clock::now() >= at; }
    double remaining() const {
        if (!bounded) return std::numeric_limits<double>::infinity();
        return std::chrono::duration<double>(at - Clock::now()).count();
    }
};

struct H128 {
    uint64_t a = 0, b = 0;
    bool operator==(const H128& o) const { return a == o.a && b == o.b; }
};
struct H128Hash {
    size_t operator()(const H128& k) const noexcept { return (size_t)(k.a ^ (k.b * 0x9E3779B97F4A7C15ull)); }
};

// Exact state interning: Zobrist hash buckets, full comparison of the sorted lists.
class StateTable {
public:
    uint32_t intern(const std::vector<uint32_t>& sorted_sel, const H128& key) {
        auto& bucket = buckets_[key];
        for (uint32_t id : bucket) {
            const uint64_t s = off_[id], e = off_[id + 1];
            if (e - s == sorted_sel.size() && std::equal(sorted_sel.begin(), sorted_sel.end(), flat_.begin() + s))
                return id;
        }
        const uint32_t id = (uint32_t)(off_.size() - 1);
        flat_.insert(flat_.end(), sorted_sel.begin(), sorted_sel.end());
        off_.push_back(flat_.size());
        bucket.push_back(id);
        return id;
    }
    size_t size() const { return off_.size() - 1; }

private:
    std::unordered_map<H128, std::vector<uint32_t>, H128Hash> buckets_;
    std::vector<uint32_t> flat_;
    std::vector<uint64_t> off_{0};
};

class Sarsa {
public:
    Sarsa(uint32_t D, const std::vector<std::pair<uint32_t, uint32_t>>& edges,
          py::array_t<uint32_t, py::array::c_style | py::array::forcecast> inst_nodes,
          py::array_t<uint64_t, py::array::c_style | py::array::forcecast> inst_off,
          py::array_t<double, py::array::c_style | py::array::forcecast> gains,
          py::array_t<int32_t, py::array::c_style | py::array::forcecast> pattern,
          py::array_t<double, py::array::c_style | py::array::forcecast> pattern_area,
          double area_budget, double area_eps)
        : D_(D), W_((D + 63u) / 64u), succ_(D), budget_(area_budget), area_eps_(area_eps) {
        for (auto& e : edges) {
            if (e.first >= D || e.second >= D) throw std::invalid_argument("edge index out of range");
            if (e.first >= e.second) throw std::invalid_argument("nodes must be in topological order");
            succ_[e.first].push_back(e.second);
        }
        for (auto& s : succ_) {
            std::sort(s.begin(), s.end());
            s.erase(std::unique(s.begin(), s.end()), s.end());
        }
        auto off = inst_off.unchecked<1>();
        auto nodes = inst_nodes.unchecked<1>();
        if (off.shape(0) < 1) throw std::invalid_argument("offsets must have at least one entry");
        N_ = (uint32_t)(off.shape(0) - 1);
        if ((uint64_t)nodes.shape(0) != off(N_)) throw std::invalid_argument("offsets do not match nodes");
        off_.resize(N_ + 1);
        for (uint32_t i = 0; i <= N_; ++i) off_[i] = off(i);
        nodes_.resize(nodes.shape(0));
        for (ssize_t i = 0; i < nodes.shape(0); ++i) {
            if (nodes(i) >= D) throw std::invalid_argument("instance node out of range");
            nodes_[i] = nodes(i);
        }
        auto g = gains.unchecked<1>();
        auto p = pattern.unchecked<1>();
        auto pa = pattern_area.unchecked<1>();
        if ((uint32_t)g.shape(0) != N_ || (uint32_t)p.shape(0) != N_)
            throw std::invalid_argument("gains/pattern must have one entry per instance");
        P_ = (uint32_t)pa.shape(0);
        gain_.resize(N_);
        pat_.resize(N_);
        for (uint32_t i = 0; i < N_; ++i) {
            gain_[i] = g(i);
            if (p(i) < 0 || (uint32_t)p(i) >= P_) throw std::invalid_argument("pattern index out of range");
            pat_[i] = (uint32_t)p(i);
        }
        parea_.resize(P_);
        for (uint32_t k = 0; k < P_; ++k) parea_[k] = pa(k);
        // strict reachability of the DDG
        reach0_.assign((size_t)D_ * W_, 0ull);
        for (uint32_t u = D_; u-- > 0;) {
            uint64_t* r = &reach0_[(size_t)u * W_];
            for (uint32_t s : succ_[u]) {
                r[s >> 6] |= 1ull << (s & 63);
                const uint64_t* rs = &reach0_[(size_t)s * W_];
                for (uint32_t w = 0; w < W_; ++w) r[w] |= rs[w];
            }
        }
        // instances whose pattern alone exceeds the budget are never actions
        for (uint32_t i = 0; i < N_; ++i)
            if (parea_[pat_[i]] <= budget_ + area_eps_) base_.push_back(i);
    }

    py::dict run(double alpha, double gamma, double eps_start, double eps_end, uint32_t episodes,
                 uint64_t seed, double seconds) {
        py::dict out;
        std::vector<uint32_t> chosen;
        uint32_t done = 0;
        bool cut_training = false, cut_rollout = false;
        double eps_last = eps_start;
        uint64_t dep_rejections = 0, steps = 0;
        size_t n_states = 0, n_q = 0;
        double train_s = 0.0, rollout_s = 0.0;
        {
            py::gil_scoped_release nogil;
            Deadline dl(seconds);
            rng_.seed(seed);
            init_zobrist(seed);
            stamp_inst_.assign(N_, 0);
            stamp_node_.assign(D_, 0);
            stamp_pat_.assign(P_, 0);
            in_c_.assign(D_, 0);
            Q_.clear();
            states_ = StateTable();
            epoch_ = 0;
            dep_rej_ = 0;
            auto t_train = Clock::now();
            for (uint32_t ep = 0; ep < episodes; ++ep) {
                const double elapsed = std::chrono::duration<double>(Clock::now() - t_train).count();
                const double mean_ep = ep ? elapsed / ep : 0.0;
                if (dl.bounded && dl.remaining() <= 2.0 * mean_ep) { cut_training = true; break; }
                if (dl.expired()) { cut_training = true; break; }
                const double frac = episodes ? (double)ep / (double)episodes : 0.0;
                const double eps = std::max(eps_end, eps_start * (1.0 - frac));
                eps_last = eps;
                bool cut = false;
                steps += episode(alpha, gamma, eps, true, dl, nullptr, cut);
                if (cut) { cut_training = true; break; }
                ++done;
            }
            train_s = std::chrono::duration<double>(Clock::now() - t_train).count();
            auto t_roll = Clock::now();
            bool cut = false;
            episode(alpha, gamma, 0.0, false, dl, &chosen, cut);
            cut_rollout = cut;
            rollout_s = std::chrono::duration<double>(Clock::now() - t_roll).count();
            dep_rejections = dep_rej_;
            n_states = states_.size();
            for (auto& q : Q_) n_q += q.size();
        }
        out["chosen"] = chosen;
        out["episodes_done"] = done;
        out["training_cut"] = cut_training;
        out["rollout_cut"] = cut_rollout;
        out["epsilon_last"] = eps_last;
        out["dependency_rejections"] = dep_rejections;
        out["steps"] = steps;
        out["states"] = n_states;
        out["q_entries"] = n_q;
        out["train_s"] = train_s;
        out["rollout_s"] = rollout_s;
        out["num_actions"] = (uint64_t)base_.size();
        return out;
    }

    uint32_t num_instances() const { return N_; }

private:
    uint32_t D_, W_, N_ = 0, P_ = 0;
    std::vector<std::vector<uint32_t>> succ_;
    std::vector<uint64_t> off_;
    std::vector<uint32_t> nodes_;
    std::vector<double> gain_;
    std::vector<uint32_t> pat_;
    std::vector<double> parea_;
    double budget_, area_eps_;
    std::vector<uint64_t> reach0_;
    std::vector<uint32_t> base_;

    // per-run
    std::mt19937_64 rng_;
    std::vector<H128> zob_;
    std::vector<std::vector<std::pair<uint32_t, double>>> Q_;  // by state id
    StateTable states_;
    uint32_t epoch_ = 0;
    uint64_t dep_rej_ = 0;

    // per-episode
    std::vector<uint32_t> stamp_inst_;  // == epoch_: excluded this episode
    std::vector<uint32_t> stamp_node_;  // == epoch_: node covered
    std::vector<uint32_t> stamp_pat_;   // == epoch_: pattern implemented
    std::vector<uint8_t> in_c_;
    std::vector<uint64_t> reach_;
    std::vector<uint32_t> avail_;
    size_t head_ = 0, n_dead_ = 0;
    double used_area_ = 0.0;
    std::vector<uint32_t> sel_sorted_;
    H128 key_{};

    void init_zobrist(uint64_t seed) {
        std::mt19937_64 g(seed ^ 0x51BADD5EFA11F00Dull);
        zob_.resize(N_);
        for (auto& z : zob_) {
            z.a = g();
            z.b = g();
            if (!z.a && !z.b) z.a = 1;
        }
    }

    double q_get(uint32_t sid, uint32_t a) const {
        if (sid >= Q_.size()) return 0.0;
        for (auto& e : Q_[sid]) if (e.first == a) return e.second;
        return 0.0;
    }
    void q_set(uint32_t sid, uint32_t a, double v) {
        if (sid >= Q_.size()) Q_.resize(sid + 1);
        for (auto& e : Q_[sid]) if (e.first == a) { e.second = v; return; }
        Q_[sid].emplace_back(a, v);
    }

    bool dead(uint32_t c) const { return stamp_inst_[c] == epoch_; }
    void kill(uint32_t c) {
        if (stamp_inst_[c] != epoch_) { stamp_inst_[c] = epoch_; ++n_dead_; }
    }

    bool dependency_cycle(uint32_t c) {
        const uint64_t s0 = off_[c], s1 = off_[c + 1];
        for (uint64_t k = s0; k < s1; ++k) in_c_[nodes_[k]] = 1;
        bool cyc = false;
        for (uint64_t k = s0; k < s1 && !cyc; ++k) {
            for (uint32_t s : succ_[nodes_[k]]) {
                if (in_c_[s]) continue;
                const uint64_t* r = &reach_[(size_t)s * W_];
                for (uint64_t j = s0; j < s1; ++j) {
                    const uint32_t y = nodes_[j];
                    if ((r[y >> 6] >> (y & 63)) & 1ull) { cyc = true; break; }
                }
                if (cyc) break;
            }
        }
        for (uint64_t k = s0; k < s1; ++k) in_c_[nodes_[k]] = 0;
        return cyc;
    }

    // action test with lazy exclusion (all relations are monotone in the selection)
    bool valid(uint32_t c) {
        if (dead(c)) return false;
        for (uint64_t k = off_[c]; k < off_[c + 1]; ++k)
            if (stamp_node_[nodes_[k]] == epoch_) { kill(c); return false; }
        const uint32_t p = pat_[c];
        if (stamp_pat_[p] != epoch_ && used_area_ + parea_[p] > budget_ + area_eps_) { kill(c); return false; }
        if (dependency_cycle(c)) { ++dep_rej_; kill(c); return false; }
        return true;
    }

    void compact() {
        size_t w = 0;
        for (size_t r = 0; r < avail_.size(); ++r)
            if (!dead(avail_[r])) avail_[w++] = avail_[r];
        avail_.resize(w);
        head_ = 0;
        n_dead_ = 0;
    }

    // first valid action in priority order (advances head_); -1 if none
    int64_t first_valid() {
        while (head_ < avail_.size()) {
            const uint32_t c = avail_[head_];
            if (valid(c)) return c;
            ++head_;
        }
        return -1;
    }

    int64_t choose(uint32_t sid, double eps) {
        const int64_t first = first_valid();
        if (first < 0) return -1;
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        if (eps > 0.0 && u01(rng_) < eps) {
            // uniform over the valid actions by rejection
            while (true) {
                if (n_dead_ * 2 > avail_.size()) compact();
                std::uniform_int_distribution<size_t> di(0, avail_.size() - 1);
                const uint32_t c = avail_[di(rng_)];
                if (valid(c)) return c;
            }
        }
        // greedy: max Q, ties to the lowest index; unvisited actions have Q = 0
        int64_t best_a = -1;
        double best_q = -std::numeric_limits<double>::infinity();
        const bool has_row = sid < Q_.size();
        if (has_row) {
            for (auto& e : Q_[sid]) {
                const bool better = e.second > best_q || (e.second == best_q && (int64_t)e.first < best_a);
                if (better && valid(e.first)) { best_q = e.second; best_a = e.first; }
            }
        }
        int64_t unvisited = -1;
        for (size_t i = head_; i < avail_.size(); ++i) {
            const uint32_t c = avail_[i];
            if (dead(c)) continue;
            bool visited = false;
            if (has_row) for (auto& e : Q_[sid]) if (e.first == c) { visited = true; break; }
            if (visited) continue;
            if (valid(c)) { unvisited = c; break; }
        }
        if (best_a < 0) return unvisited;
        if (unvisited < 0) return best_a;
        if (best_q > 0.0) return best_a;
        if (best_q < 0.0) return unvisited;
        return std::min(best_a, unvisited);
    }

    void apply(uint32_t a) {
        const uint64_t s0 = off_[a], s1 = off_[a + 1];
        for (uint64_t k = s0; k < s1; ++k) stamp_node_[nodes_[k]] = epoch_;
        const uint32_t p = pat_[a];
        if (stamp_pat_[p] != epoch_) { stamp_pat_[p] = epoch_; used_area_ += parea_[p]; }
        sel_sorted_.insert(std::lower_bound(sel_sorted_.begin(), sel_sorted_.end(), a), a);
        key_.a ^= zob_[a].a;
        key_.b ^= zob_[a].b;
        kill(a);
        // contracted reachability: everything reaching `a` (and `a` itself) reaches `a` and its closure
        std::vector<uint64_t> down(W_, 0ull);
        for (uint64_t k = s0; k < s1; ++k) {
            const uint32_t x = nodes_[k];
            down[x >> 6] |= 1ull << (x & 63);
            const uint64_t* r = &reach_[(size_t)x * W_];
            for (uint32_t w = 0; w < W_; ++w) down[w] |= r[w];
        }
        for (uint64_t k = s0; k < s1; ++k) in_c_[nodes_[k]] = 1;
        for (uint32_t u = 0; u < D_; ++u) {
            uint64_t* r = &reach_[(size_t)u * W_];
            bool hit = in_c_[u] != 0;
            for (uint64_t k = s0; k < s1 && !hit; ++k) {
                const uint32_t y = nodes_[k];
                hit = (r[y >> 6] >> (y & 63)) & 1ull;
            }
            if (hit) for (uint32_t w = 0; w < W_; ++w) r[w] |= down[w];
        }
        for (uint64_t k = s0; k < s1; ++k) in_c_[nodes_[k]] = 0;
    }

    void reset_episode() {
        if (++epoch_ == 0) {  // wrap: clear stamps
            std::fill(stamp_inst_.begin(), stamp_inst_.end(), 0);
            std::fill(stamp_node_.begin(), stamp_node_.end(), 0);
            std::fill(stamp_pat_.begin(), stamp_pat_.end(), 0);
            epoch_ = 1;
        }
        reach_ = reach0_;
        avail_ = base_;
        head_ = 0;
        n_dead_ = 0;
        used_area_ = 0.0;
        sel_sorted_.clear();
        key_ = H128{};
    }

    // One episode; learn=false is the greedy rollout that records the selection.
    uint64_t episode(double alpha, double gamma, double eps, bool learn, const Deadline& dl,
                     std::vector<uint32_t>* chosen, bool& cut) {
        reset_episode();
        uint64_t n = 0;
        uint32_t sid = states_.intern(sel_sorted_, key_);
        int64_t a = choose(sid, eps);
        while (a >= 0) {
            if ((n & 15u) == 0u && dl.expired()) { cut = true; break; }
            const uint32_t act = (uint32_t)a;
            const double r = gain_[act];
            apply(act);
            if (chosen) chosen->push_back(act);
            ++n;
            const uint32_t sid2 = states_.intern(sel_sorted_, key_);
            const int64_t a2 = choose(sid2, eps);
            if (learn) {
                const double target = (a2 < 0) ? r : r + gamma * q_get(sid2, (uint32_t)a2);
                const double q = q_get(sid, act);
                q_set(sid, act, q + alpha * (target - q));
            }
            sid = sid2;
            a = a2;
        }
        return n;
    }
};

}  // namespace

PYBIND11_MODULE(_sarsa_core, m) {
    m.doc() = "Tabular SARSA CI selection with exact conflict handling (ci_ident.baselines.sarsa)";
    py::class_<Sarsa>(m, "Sarsa")
        .def(py::init<uint32_t, const std::vector<std::pair<uint32_t, uint32_t>>&,
                      py::array_t<uint32_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<uint64_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>,
                      py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
                      py::array_t<double, py::array::c_style | py::array::forcecast>, double, double>(),
             py::arg("num_nodes"), py::arg("edges"), py::arg("inst_nodes"), py::arg("inst_offsets"),
             py::arg("gains"), py::arg("pattern"), py::arg("pattern_area"), py::arg("area_budget"),
             py::arg("area_eps"))
        .def("run", &Sarsa::run, py::arg("alpha"), py::arg("gamma"), py::arg("eps_start"),
             py::arg("eps_end"), py::arg("episodes"), py::arg("seed"), py::arg("seconds"))
        .def_property_readonly("num_instances", &Sarsa::num_instances);
    m.attr("ABI_VERSION") = 1;
}
