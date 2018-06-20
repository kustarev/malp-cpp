#ifndef malp_range_h
#define malp_range_h

#include <map>

namespace {
    
    class Range
    {
        std::vector<size_t> n_;
    public:
        Range(size_t n) : n_(n)
        {
            for (size_t i = 0; i < n; i++)
                n_[i] = i;
        }
        const std::vector<size_t>::const_iterator begin() const
        {
            return n_.begin();
        }
        const std::vector<size_t>::const_iterator end() const
        {
            return n_.end();
        }
        const std::vector<size_t>& vector() const
        {
            return n_;
        }
    };
    
    class Ranges
    {
        std::map<size_t, Range> m_;
        Ranges() {}
        void update(size_t n)
        {
            if (m_.find(n) == m_.end())
                m_.insert(std::pair<size_t, Range>(n, Range(n)));
        }
    public:
        Ranges(Ranges const&) = delete;
        void operator=(Ranges const&) = delete;
        static const Range& get(size_t n)
        {
            static Ranges instance;
            instance.update(n);
            return instance.m_.find(n) -> second;
        }
    };
    
}

const Range& range(size_t n)
{
    return Ranges::get(n);
}

#endif
