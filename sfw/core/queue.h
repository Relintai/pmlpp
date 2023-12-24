#ifndef QUEUE_H
#define QUEUE_H

#include "core/error_macros.h"

#include "vector.h"

template<class T>
class Queue
{
public:
    void enqueue(const T &job);
    T dequeue();
    const T &peek() const;

    bool is_empty() const;
    bool is_full() const;

    int size() const;
    int capacity() const;
    void ensure_capacity(const int capacity);
    void resize(const int s);

    void pack();

    Queue();
    Queue(int prealloc);
    Queue(int prealloc, int grow_by);
    ~Queue();

private:
    T* _data;
    int _head;
    int _tail;

    int _capacity;
    int _grow_size;
};

template <class T>
void Queue<T>::pack()
{
    if (_head == _tail)
    {
        _head = 0;
        _tail = 0;
        return;
    }

    if (_head == 0 && _tail == _capacity)
    {
        resize(_capacity + _grow_size);
        return;
    }

    int j = 0;

    for (int i = _head; i < _tail; ++i)
    {
        _data[j++] = _data[i];
    }

    _head = 0;
    _tail = j;

}

template <class T>
void Queue<T>::enqueue(const T &job)
{
    if (_tail == _capacity)
    {
        pack();
    }

    _data[_tail++] = job;
}

template <class T>
T Queue<T>::dequeue()
{
    CRASH_COND(is_empty());

    return _data[_head++];
}

template <class T>
const T &Queue<T>::peek() const
{
    CRASH_COND(is_empty());

    return _data[_head];
}

template <class T>
bool Queue<T>::is_empty() const
{
    return _head == _tail;
}

template <class T>
bool Queue<T>::is_full() const
{
    return (_head == 0) && (_tail == _capacity);
}

template <class T>
int Queue<T>::size() const
{
    return _tail - _head;
}

template <class T>
int Queue<T>::capacity() const
{
    return _capacity;
}

template <class T>
void Queue<T>::ensure_capacity(const int capacity)
{
    if (_capacity < capacity)
    {
        resize(capacity);
    }
}

template <class T>
void Queue<T>::resize(const int s)
{
    if (s == 0 && _data)
    {
        delete[] _data;

        _data = nullptr;

        _capacity = s;

        return;
    }

    if (!_data)
    {
        _data = new T[s];
        _capacity = s;
        return;
    }

    if (is_empty())
    {
        delete[] _data;

        _data = new T[s];

        _capacity = s;

        return;
    }

    if (!is_full())
    {
        pack();
    }

    T* nd = new T[s];

    int to = s > _tail ? _tail : s;

    for (int i = 0; i < to; ++i)
    {
        nd[i] = _data[i];
    }

    delete[] _data;
    _data = nd;
    _capacity = s;
}

template <class T>
Queue<T>::Queue()
{
    _data = nullptr;

    _head = 0;
    _tail = 0;

    _capacity = 20;
    _grow_size = 10;
    resize(_capacity);
}

template <class T>
Queue<T>::Queue(int prealloc)
{
    _data = nullptr;

    _head = 0;
    _tail = 0;

    _capacity = prealloc;
    _grow_size = 10;
    resize(_capacity);
}

template <class T>
Queue<T>::Queue(int prealloc, int grow_by)
{
    _data = nullptr;

    _head = 0;
    _tail = 0;

    _capacity = prealloc;
    _grow_size = grow_by;
    resize(_capacity);
}

template <class T>
Queue<T>::~Queue()
{
    if (_data)
    {
        delete[] _data;

        _data = nullptr;
    }
}

#endif
