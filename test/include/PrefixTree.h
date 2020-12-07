#pragma once
#include <cassert>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <vector>

class PrefixTreeNode
{
public:
    template <class Iterator>
    void add( Iterator begin, Iterator end )
    {
        total_children++;
        if( begin == end )
        {
            leaf = true;
            return;
        }
        assert( !leaf );
        auto child_it = getOrInsert<std::unique_ptr<PrefixTreeNode>>( children, *begin,
                                                                      &std::make_unique<PrefixTreeNode> );
        child_it->second->add( begin + 1, end );
    }

    uint64_t countDistinct()
    {
        if( leaf )
            return 1;
        uint64_t count = 0;
        for( auto& child : children )
            count += child.second->countDistinct();
        return count;
    }

    uint64_t countDuplicates()
    {
        if( leaf )
            return total_children > 1 ? total_children : 0;
        uint64_t count = 0;
        for( auto& child : children )
            count += child.second->countDuplicates();
        return count;
    }

    std::vector<std::map<uint64_t, uint64_t>> frequencyPerLevel()
    {
        std::vector<std::map<uint64_t, uint64_t>> res;
        frequencyPerLevel( res, 0 );
        return res;
    }

    uint64_t getTotalChildren()
    {
        return total_children;
    }

private:
    template <class Value>
    auto getOrInsert( std::map<uint64_t, Value>& map, uint64_t key, std::function<Value()> gen )
    {
        auto child_it = map.lower_bound( key );
        if( child_it == map.end() || child_it->first != key )
        {
            child_it = map.emplace_hint( child_it, key, gen() );
        }
        return child_it;
    }

    void frequencyPerLevel( std::vector<std::map<uint64_t, uint64_t>>& vec, uint64_t depth )
    {
        if( leaf )
            return;


        if( vec.size() <= depth )
        {
            vec.emplace_back();
        }

        for( auto& child : children )
        {
            auto it = getOrInsert<uint64_t>( vec[depth], child.first, []() -> uint64_t { return 0; } );
            it->second += child.second->getTotalChildren();
            child.second->frequencyPerLevel( vec, depth + 1 );
        }
    }

    uint64_t total_children = 0;
    bool leaf = false;
    std::map<uint64_t, std::unique_ptr<PrefixTreeNode>> children;
};

using PrefixTree = PrefixTreeNode;
