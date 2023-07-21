#ifndef FIXEDQUEUE_H
#define FIXEDQUEUE_H

#include <queue>
#include <deque>
#include <iostream>

// Adapted from https://stackoverflow.com/a/56334648
template <typename T, int MaxLen, typename Container = std::deque<T>>
class FixedQueue : public std::queue<T, Container> {
public:
    void push(const T& value) {
        if (this->size() == MaxLen) {
            this->c.pop_front();
        }
        std::queue<T, Container>::push(value);
    }
};

#endif