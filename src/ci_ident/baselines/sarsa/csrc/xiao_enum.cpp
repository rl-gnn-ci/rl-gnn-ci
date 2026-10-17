// Exact enumeration of convex, I/O-bounded (optionally connected) CI
// candidates of a DDG.  Python module: ci_ident.baselines.sarsa._xiao_enum
//
// Search (after Xiao et al., 2014): depth-first growth of a convex node set M
// from singleton roots.  The children of M are the NodeFilter frontier
//   IPred(M) nodes n with Succ*(n) and IPred(M) disjoint   (origin "pred")
//   ISucc(M) nodes n with Pred*(n) and ISucc(M) disjoint   (origin "succ")
//   [connected == false, |Out(M)| < OUTmax]
//   nodes with no path to or from M                        (origin "disc")
// in ascending topological index.  Each such extension keeps M convex.  After
// a child v has been explored, sibling deletion adds v and Pred*(v) (origin
// pred), v and Succ*(v) (origin succ) or v alone (origin disc) to the
// exclusion set R of the remaining siblings: every convex superset of M that
// contains a node of Pred*(v) (resp. Succ*(v)) also contains v and has been
// generated in v's subtree.  The roots are processed in topological order and
// each processed root is added to the exclusion set of the later roots (the
// same rule at the top level), so every candidate is generated exactly once,
// from its topologically first node.
//
// Pruning is by permanent I/O only: an outside predecessor of M that is
// inadmissible or excluded is an input of every set in the subtree, and a
// node of M that is a DDG sink or has an inadmissible or excluded successor
// is an output of every set in the subtree.  A state is expanded whether or
// not its own I/O counts are within the bounds, so the search is exhaustive.
//
// I/O semantics: In(S) = distinct nodes outside S with a successor in S;
// Out(S) = nodes of S with a successor outside S or with no successor.
//
// For each candidate the longest hardware-latency path inside S is computed
// (dist(v) = hw(v) + max over predecessors p in S of dist(p)); candidates
// with |S| < 2 or |S| - ceil(cp - eps) <= 0 are counted but, unless
// store_all is set, not stored.
//
// The deadline is a duration in seconds measured from the call on the
// monotonic clock: +inf means no limit, a value <= 0 is already expired.
// max_stored > 0 bounds the number of stored candidates (and so the memory of
// the result); the search stops when the bound is reached (capped = true).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

namespace {

template <class T>
py::array_t<T> to_numpy(std::vector<T>&& v) {
    auto* owned = new std::vector<T>(std::move(v));
    py::capsule cap(owned, [](void* p) { delete reinterpret_cast<std::vector<T>*>(p); });
    return py::array_t<T>((py::ssize_t)owned->size(), owned->data(), cap);
}

struct Deadline {
    bool bounded = false;
    Clock::time_point at{};
    explicit Deadline(double seconds) {
        if (std::isfinite(seconds)) {
            bounded = true;
            double s = seconds > 0.0 ? seconds : 0.0;
            at = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                    std::chrono::duration<double>(s));
        }
    }
    bool expired() const { return bounded && Clock::now() >= at; }
};

class Enumerator {
public:
    Enumerator(uint32_t n, const std::vector<std::pair<uint32_t, uint32_t>>& edges,
               const std::vector<uint8_t>& eligible, const std::vector<double>& hw)
        : N(n), W((n + 63u) / 64u), pred(n), succ(n), forbidden(n), sink(n), hw_(hw) {
        if (eligible.size() != N || hw.size() != N)
            throw std::invalid_argument("eligible/hw must have one entry per node");
        for (auto& e : edges) {
            if (e.first >= N || e.second >= N) throw std::invalid_argument("edge index out of range");
            if (e.first >= e.second)
                throw std::invalid_argument("nodes must be given in topological order (u < v for every edge)");
            succ[e.first].push_back(e.second);
            pred[e.second].push_back(e.first);
        }
        for (uint32_t i = 0; i < N; ++i) {
            auto dedup = [](std::vector<uint32_t>& v) {
                std::sort(v.begin(), v.end());
                v.erase(std::unique(v.begin(), v.end()), v.end());
            };
            dedup(pred[i]);
            dedup(succ[i]);
            forbidden[i] = eligible[i] ? 0 : 1;
            sink[i] = succ[i].empty() ? 1 : 0;
        }
        // transitive closures (strict) as bitsets; indices are topological
        desc.assign((size_t)N * W, 0ull);
        anc.assign((size_t)N * W, 0ull);
        for (uint32_t k = N; k-- > 0;) {
            uint64_t* d = &desc[(size_t)k * W];
            for (uint32_t s : succ[k]) {
                d[s >> 6] |= 1ull << (s & 63);
                const uint64_t* ds = &desc[(size_t)s * W];
                for (uint32_t w = 0; w < W; ++w) d[w] |= ds[w];
            }
        }
        for (uint32_t k = 0; k < N; ++k) {
            uint64_t* a = &anc[(size_t)k * W];
            for (uint32_t p : pred[k]) {
                a[p >> 6] |= 1ull << (p & 63);
                const uint64_t* ap = &anc[(size_t)p * W];
                for (uint32_t w = 0; w < W; ++w) a[w] |= ap[w];
            }
        }
    }

    py::dict run(int in_max, int out_max, bool connected, double seconds, bool store_all, double eps,
                 uint64_t max_stored) {
        IN = in_max;
        OUT = out_max;
        connected_ = connected;
        store_all_ = store_all;
        eps_ = eps;
        max_stored_ = max_stored;
        capped = false;
        res_nodes.clear();
        res_off.assign(1, 0);
        res_cp.clear();
        explored = valid = 0;
        stop = false;
        inM.assign(N, 0);
        Mbits.assign(W, 0ull);
        members.clear();
        stampP.assign(N, 0);
        stampS.assign(N, 0);
        stamp = 0;
        dist.assign(N, 0.0);
        auto t0 = Clock::now();
        {
            py::gil_scoped_release nogil;
            Deadline dl(seconds);
            dl_ = &dl;
            if (dl.expired()) stop = true;
            std::vector<uint64_t> R0(W, 0ull);
            for (uint32_t u = 0; u < N && !stop; ++u) {
                if (forbidden[u]) continue;
                add(u);
                std::vector<uint64_t> R = R0;
                dfs(R);
                remove(u);
                R0[u >> 6] |= 1ull << (u & 63);
            }
            dl_ = nullptr;
        }
        double elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
        py::dict out;
        const uint64_t stored = (uint64_t)res_cp.size();
        out["nodes"] = to_numpy(std::move(res_nodes));  // ownership moves to the arrays
        out["offsets"] = to_numpy(std::move(res_off));
        out["cp"] = to_numpy(std::move(res_cp));
        res_nodes = {};
        res_off = {};
        res_cp = {};
        out["explored"] = explored;
        out["valid"] = valid;
        out["stored"] = stored;
        out["timed_out"] = stop && !capped;
        out["capped"] = capped;
        out["elapsed_s"] = elapsed;
        return out;
    }

    uint32_t num_nodes() const { return N; }

private:
    uint32_t N, W;
    std::vector<std::vector<uint32_t>> pred, succ;
    std::vector<uint8_t> forbidden, sink;
    std::vector<double> hw_;
    std::vector<uint64_t> desc, anc;

    int IN = 0, OUT = 0;
    bool connected_ = true, store_all_ = false;
    double eps_ = 1e-9;
    const Deadline* dl_ = nullptr;
    bool stop = false, capped = false;
    uint64_t max_stored_ = 0;
    uint64_t explored = 0, valid = 0;

    std::vector<uint8_t> inM;
    std::vector<uint64_t> Mbits;
    std::vector<uint32_t> members;
    std::vector<uint32_t> stampP, stampS;
    uint32_t stamp = 0;
    std::vector<double> dist;

    std::vector<uint32_t> res_nodes;
    std::vector<uint64_t> res_off;
    std::vector<double> res_cp;

    static bool test(const std::vector<uint64_t>& b, uint32_t i) { return (b[i >> 6] >> (i & 63)) & 1ull; }
    bool test_row(const std::vector<uint64_t>& m, uint32_t row, uint32_t i) const {
        return (m[(size_t)row * W + (i >> 6)] >> (i & 63)) & 1ull;
    }
    void add(uint32_t v) {
        inM[v] = 1;
        Mbits[v >> 6] |= 1ull << (v & 63);
        members.push_back(v);
    }
    void remove(uint32_t v) {
        inM[v] = 0;
        Mbits[v >> 6] &= ~(1ull << (v & 63));
        members.pop_back();
    }
    uint32_t next_stamp() {
        if (++stamp == 0) {
            std::fill(stampP.begin(), stampP.end(), 0);
            std::fill(stampS.begin(), stampS.end(), 0);
            stamp = 1;
        }
        return stamp;
    }

    void record() {
        ++valid;
        std::vector<uint32_t> S(members);
        std::sort(S.begin(), S.end());  // topological
        double cp = 0.0;
        for (uint32_t v : S) {
            double base = 0.0;
            for (uint32_t p : pred[v])
                if (inM[p] && dist[p] > base) base = dist[p];
            dist[v] = hw_[v] + base;
            if (dist[v] > cp) cp = dist[v];
        }
        if (!store_all_) {
            if (S.size() < 2) return;
            double saving = (double)S.size() - std::ceil(cp - eps_);
            if (!(saving > 0.0)) return;
        }
        res_nodes.insert(res_nodes.end(), S.begin(), S.end());
        res_off.push_back(res_nodes.size());
        res_cp.push_back(cp);
        if (max_stored_ && (uint64_t)res_cp.size() >= max_stored_) {
            capped = true;
            stop = true;
        }
    }

    void sibling_delete(uint32_t v, int origin, std::vector<uint64_t>& R) const {
        R[v >> 6] |= 1ull << (v & 63);
        if (origin == 0) {
            const uint64_t* a = &anc[(size_t)v * W];
            for (uint32_t w = 0; w < W; ++w) R[w] |= a[w];
        } else if (origin == 1) {
            const uint64_t* d = &desc[(size_t)v * W];
            for (uint32_t w = 0; w < W; ++w) R[w] |= d[w];
        }
    }

    void dfs(std::vector<uint64_t>& R) {
        if (stop) return;
        if ((++explored & 1023u) == 0u && dl_->expired()) {
            stop = true;
            return;
        }
        // I/O counts and permanent I/O
        const uint32_t sp = next_stamp();
        std::vector<uint32_t> ipred, isucc;
        uint32_t n_in = 0, perm_in = 0, n_out = 0, perm_out = 0;
        for (uint32_t u : members) {
            for (uint32_t p : pred[u]) {
                if (inM[p] || stampP[p] == sp) continue;
                stampP[p] = sp;
                ++n_in;
                ipred.push_back(p);
                if (forbidden[p] || test(R, p)) ++perm_in;
            }
            bool is_out = sink[u] != 0, perm = sink[u] != 0;
            for (uint32_t s : succ[u]) {
                if (inM[s]) continue;
                is_out = true;
                if (forbidden[s] || test(R, s)) perm = true;
                if (stampS[s] != sp) {
                    stampS[s] = sp;
                    isucc.push_back(s);
                }
            }
            n_out += is_out;
            perm_out += perm;
        }
        if ((int)perm_in > IN || (int)perm_out > OUT) return;
        if ((int)n_in <= IN && (int)n_out <= OUT) record();

        // NodeFilter frontier (origin 0 = pred, 1 = succ, 2 = disc)
        std::vector<std::pair<uint32_t, int>> frontier;
        frontier.reserve(ipred.size() + isucc.size());
        for (uint32_t n : ipred) {
            bool ok = true;
            for (uint32_t w : ipred)
                if (w != n && test_row(desc, n, w)) { ok = false; break; }
            if (ok) frontier.emplace_back(n, 0);
        }
        for (uint32_t n : isucc) {
            bool ok = true;
            for (uint32_t w : isucc)
                if (w != n && test_row(anc, n, w)) { ok = false; break; }
            if (ok) frontier.emplace_back(n, 1);
        }
        if (!connected_ && (int)n_out < OUT) {
            for (uint32_t v = 0; v < N; ++v) {
                if (inM[v] || forbidden[v] || test(R, v)) continue;
                const uint64_t* a = &anc[(size_t)v * W];
                const uint64_t* d = &desc[(size_t)v * W];
                bool related = false;
                for (uint32_t w = 0; w < W && !related; ++w) related = ((a[w] | d[w]) & Mbits[w]) != 0;
                if (!related) frontier.emplace_back(v, 2);
            }
        }
        std::sort(frontier.begin(), frontier.end());

        for (const auto& c : frontier) {
            if (stop) return;
            const uint32_t v = c.first;
            if (!(forbidden[v] || test(R, v))) {
                add(v);
                std::vector<uint64_t> childR(R);
                dfs(childR);
                remove(v);
            }
            sibling_delete(v, c.second, R);
        }
    }
};

}  // namespace

PYBIND11_MODULE(_xiao_enum, m) {
    m.doc() = "Exact convex, I/O-bounded CI candidate enumeration (ci_ident.baselines.sarsa)";
    py::class_<Enumerator>(m, "Enumerator")
        .def(py::init<uint32_t, const std::vector<std::pair<uint32_t, uint32_t>>&,
                      const std::vector<uint8_t>&, const std::vector<double>&>(),
             py::arg("num_nodes"), py::arg("edges"), py::arg("eligible"), py::arg("hw_latency"))
        .def("run", &Enumerator::run, py::arg("max_in"), py::arg("max_out"), py::arg("connected"),
             py::arg("seconds"), py::arg("store_all"), py::arg("eps"), py::arg("max_stored"))
        .def_property_readonly("num_nodes", &Enumerator::num_nodes);
    m.attr("ABI_VERSION") = 2;
}
