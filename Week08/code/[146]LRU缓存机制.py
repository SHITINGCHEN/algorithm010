# 运用你所掌握的数据结构，设计和实现一个 LRU (最近最少使用) 缓存机制。它应该支持以下操作： 获取数据 get 和 写入数据 put 。 
# 
#  获取数据 get(key) - 如果关键字 (key) 存在于缓存中，则获取关键字的值（总是正数），否则返回 -1。 
# 写入数据 put(key, value) - 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字/值」。当缓存容量达到上限时，它应该在
# 写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。 
# 
#  
# 
#  进阶: 
# 
#  你是否可以在 O(1) 时间复杂度内完成这两种操作？ 
# 
#  
# 
#  示例: 
# 
#  LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );
# 
# cache.put(1, 1);
# cache.put(2, 2);
# cache.get(1);       // 返回  1
# cache.put(3, 3);    // 该操作会使得关键字 2 作废
# cache.get(2);       // 返回 -1 (未找到)
# cache.put(4, 4);    // 该操作会使得关键字 1 作废
# cache.get(1);       // 返回 -1 (未找到)
# cache.get(3);       // 返回  3
# cache.get(4);       // 返回  4
#  
#  Related Topics 设计


# leetcode submit region begin(Prohibit modification and deletion)

# 使用哈希+双链表
class DoblueLinkedNode:
    def __init__(self, key = 0, val = 0):
        self.key = key
        self.value = val
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity: int):
        self.cache = dict()
        # 加入伪头部和伪尾部
        self.head = DoblueLinkedNode()
        self.tail = DoblueLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.remain = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache: return -1
        node = self.cache[key]
        self.moveToHead(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
        else:
            node = DoblueLinkedNode(key, value)
            self.addToHead(node)
            self.cache[key] = node
            if self.remain == 0:
                removed = self.removeTail()
                self.cache.pop(removed.key)
            else:
                self.remain -= 1

    def moveToHead(self, node):
        self.removeNode(node)
        self.addToHead(node)

    def addToHead(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def removeTail(self):
        node = self.tail.prev
        self.removeNode(node)
        return node

    def removeNode(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev


# 使用OrderDict
# from collections import OrderedDict
# class LRUCache:
#
#     def __init__(self, capacity: int):
#         self.dic = OrderedDict()
#         self.remain = capacity
#
#     def get(self, key: int) -> int:
#         if key not in self.dic: return -1
#         val = self.dic.pop(key)
#         # keep key as the newest one
#         self.dic[key] = val
#         return val
#
#     def put(self, key: int, value: int) -> None:
#         if key in self.dic:
#             self.dic.pop(key)
#         else:
#             if self.remain:
#                 self.remain -= 1
#             else:
#                 self.dic.popitem(last = False)
#         self.dic[key] = value




# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)
# leetcode submit region end(Prohibit modification and deletion)
