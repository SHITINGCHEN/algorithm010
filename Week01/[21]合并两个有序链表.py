# 将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。 
# 
#  
# 
#  示例： 
# 
#  输入：1->2->4, 1->3->4
# 输出：1->1->2->3->4->4
#  
#  Related Topics 链表


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        # if not l1:
        #     return l2
        # if not l2:
        #     return l1
        # head = ListNode(-1)
        # node = head
        # l1_node, l2_node = l1, l2
        # while l1_node and l2_node:
        #     if l1_node.val <= l2_node.val:
        #         node.next = l1_node
        #         node, l1_node = node.next, l1_node.next
        #     else:
        #         node.next = l2_node
        #         node, l2_node = node.next, l2_node.next
        # if l1_node:
        #     node.next = l1_node
        #     node = node.next
        # if l2_node:
        #     node.next = l2_node
        #     node = node.next
        # return head.next
        head = ListNode(-1)
        node = head
        while l1 and l2:
            if l1.val <= l2.val:
                node.next, l1 = l1, l1.next
            else:
                node.next, l2 = l2, l2.next
            node = node.next
        node.next = l1 if l1 else l2
        return head.next


# leetcode submit region end(Prohibit modification and deletion)
