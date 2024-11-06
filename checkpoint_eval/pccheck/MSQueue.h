#ifndef MS_QUEUE_H_
#define MS_QUEUE_H_

#include <atomic>
#include <stddef.h> //for null
#include <climits>

//=============================Start MSQueue Class==========================//
/* This queue is Michael and Scott's queue from DISC 1996 which is the baseline
 * of the java.util.concurrent librraty. It is not-persistent and is the
 * baseline of all its durable versions. This version DOES NOT contain any
 * memory management.
 */

template <class T>
class MSQueue
{

public:
    //====================Start Node Class==========================//
    /* Node is the type of the elements that will be in the queue.
     * It contains the following fields:
     * value     - can be of any type. It holds the data of the element.
     * next      - a pointer to the next element in the queue.
     */
    class Node
    {
    public:
        T value;
        std::atomic<Node *> next;
        Node(T val) : value(val), next(nullptr) {}
        Node() : value(T()), next(nullptr) {}
    };
    //====================End Node Class==========================//

    MSQueue() { head = tail = new Node(INT_MAX); }

    //-------------------------------------------------------------------------

    /* Enqueues a node to the queue with the given value. */
    void enq(T value)
    {
        Node *node = new Node(value);
        while (true)
        {
            Node *last = tail.load();
            Node *next = last->next.load();
            if (last == tail.load())
            {
                // not necessary but checks again before try
                if (next == nullptr)
                {
                    if (last->next.compare_exchange_strong(next, node))
                    {
                        tail.compare_exchange_strong(last, node);
                        return;
                    }
                }
                else
                {
                    tail.compare_exchange_strong(last, next);
                }
            }
        }
    }

    //-------------------------------------------------------------------------

    /* Tries to dequeue a node. Returns the value of the removed node.
     * If the queue is empty, it returns INT_MIN which symbols an
     * empty queue.
     */
    T deq()
    {
        while (true)
        {
            Node *first = head.load();
            Node *last = tail.load();
            Node *next = first->next.load();
            if (first == head.load())
            {
                if (first == last)
                {
                    if (next == nullptr)
                    {
                        return INT_MIN;
                    }
                    tail.compare_exchange_strong(last, next);
                }
                else
                {
                    T value = next->value;
                    if (head.compare_exchange_strong(first, next))
                    {
                        return value;
                    }
                }
            }
        }
    }

    //-------------------------------------------------------------------------

private:
    std::atomic<Node *> head;
    int padding[512];
    std::atomic<Node *> tail;
};

#endif /* MS_QUEUE_H_ */
