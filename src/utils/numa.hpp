#include <numa.h>
#include <set>
#include <vector>

inline
std::vector<int> numa_nodes() {
    std::set<int> nodes;
    for (int i = 0; i < numa_num_configured_cpus(); ++i)
    {
      nodes.insert(numa_node_of_cpu(i));
    }
    assert(nodes.size() >= 1);
    std::vector<int> nodes2;
    for (const auto &i : nodes)
    {
      nodes2.push_back(i);
    }
    return nodes2;
}