# 根据一棵树的前序遍历与中序遍历构造二叉树。 
# 
#  注意: 
# 你可以假设树中没有重复的元素。 
# 
#  例如，给出 
# 
#  前序遍历 preorder = [3,9,20,15,7]
# 中序遍历 inorder = [9,3,15,20,7] 
# 
#  返回如下的二叉树： 
# 
#      3
#    / \
#   9  20
#     /  \
#    15   7 
#  Related Topics 树 深度优先搜索 数组


# leetcode submit region begin(Prohibit modification and deletion)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 前序遍历结果[根，左子树，右子树]
        # 中序遍历 [左子树，根，右子树]
        # 在中序结果中定位到根，分开左子树和右子树
        # 递归地构造左子树和右子树
        # 使用dict存储元素-位置对，加快定位效率
        # index = {element: i for i, element in enumerate(inorder)}
        # def subBulidTree(preorder_left, preorder_right, inorder_left, inorder_right):
        #     # preorder_left: 在前序结果数组中开始的位置，preorder_right在前序结果数组中结束的位置
        #     if preorder_left > preorder_right:
        #         return None
        #     preorder_root = preorder_left
        #     # 定位中序结果中的根节点
        #     inorder_root = index[preorder[preorder_root]]
        #     root = TreeNode(preorder[preorder_root])
        #     size_left_tree = inorder_root - inorder_left
        #     root.left = subBulidTree(preorder_left + 1, preorder_left + size_left_tree, inorder_left, inorder_root - 1)
        #     root.right = subBulidTree(preorder_left + size_left_tree + 1, preorder_right, inorder_root + 1, inorder_right)
        #     return root
        # return subBulidTree(0, len(preorder) - 1, 0, len(inorder) - 1)


        # if inorder:
        #     ind = inorder.index(preorder.pop(0))
        #     root = TreeNode(inorder[ind])
        #     root.left = self.buildTree(preorder, inorder[0:ind])
        #     root.right = self.buildTree(preorder, inorder[ind+1:])
        #     return root
        from collections import deque
        preorder = deque(preorder)
        inord_dict = {ele : i for i, ele in enumerate(inorder)}
        def help(start, end):
            if start > end:
                return None
            root_idx = inord_dict[preorder.popleft()]
            root = TreeNode(inorder[root_idx])
            root.left = help(start, root_idx - 1)
            root.right = help(root_idx + 1, end)
            return root
        return help(0, len(inorder) - 1)

# leetcode submit region end(Prohibit modification and deletion)
