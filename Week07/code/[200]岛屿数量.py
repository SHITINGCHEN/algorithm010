# 给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。 
# 
#  岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。 
# 
#  此外，你可以假设该网格的四条边均被水包围。 
# 
#  
# 
#  示例 1: 
# 
#  输入:
# 11110
# 11010
# 11000
# 00000
# 输出: 1
#  
# 
#  示例 2: 
# 
#  输入:
# 11000
# 11000
# 00100
# 00011
# 输出: 3
# 解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。
#  
#  Related Topics 深度优先搜索 广度优先搜索 并查集


# leetcode submit region begin(Prohibit modification and deletion)
from collections import deque
class UnionFind:
    def __init__(self, n):
        self.count = n
        self.parent = [i for i in range(n)]
        self.rank = [1 for _ in range(n)] # rank的用处？

    def get_count(self):
        return self.count

    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def is_connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)
        if p_root == q_root:
            return
        # 谁的rank较高就以谁为parent
        if self.rank[p_root] > self.rank[q_root]:
            self.parent[q_root] = p_root
        elif self.rank[p_root] < self.rank[q_root]:
            self.parent[p_root] = q_root
        else:
            self.parent[q_root] = p_root
            self.rank[p_root] += 1
        self.count -= 1

class Solution:
    # 使用并查集
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid or not grid[0]: return 0
        m, n = len(grid), len(grid[0])
        directions = [(1, 0), (0, 1)]
        dummy_node = m * n # 用来存放'0'
        uf = UnionFind(dummy_node + 1)
        def get_index(x, y):
            return x * n + y
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '0':
                    uf.union(get_index(i, j), dummy_node)
                if grid[i][j] == '1':
                    for (dx, dy) in directions:
                        x, y = i + dx, j + dy
                        if x < m and y < n and grid[x][y] == '1':
                            uf.union(get_index(i, j), get_index(x, y))

        return uf.get_count() - 1

# class Solution:

    # BFS，使用辅助队列来记录相邻结点，并使用额外数组来记录已访问过的结点
    # 一旦有新的结点入队必须标记已访问，以防重复访问

    # def numIslands(self, grid: List[List[str]]) -> int:
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #     m = len(grid) # 行
    #     if m == 0:
    #         return 0
    #     n = len(grid[0]) # 列
    #     que = deque([])
    #     marked = [[False for _ in range(n)] for _ in range(m)]
    #     count = 0
    #     for i in range(m):
    #         for j in range(n):
    #             # 遇到未访问过的陆地，使用BFS标记相连的陆地
    #             if grid[i][j] == '1' and not marked[i][j]:
    #                 que.append((i, j))
    #                 marked[i][j] = True
    #                 while que:
    #                     x, y = que.popleft()
    #                     for dx, dy in directions:
    #                         new_x, new_y = x + dx, y + dy
    #                         if 0 <= new_x < m and 0 <= new_y < n and grid[new_x][new_y] == '1' and not marked[new_x][new_y]:
    #                             que.append((new_x, new_y))
    #                             marked[new_x][new_y] = True
    #
    #                 count += 1
    #     return count


    # DFS，并将已访问过的结点标为0
    # def numIslands(self, grid: List[List[str]]) -> int:
    #     def dfs(i, j):
    #         for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    #             x, y = dx + i, dy + j
    #             if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
    #                 grid[x][y] = '0'
    #                 dfs(x, y)
    #
    #     m = len(grid)
    #     if m == 0:
    #         return 0
    #     n = len(grid[0])
    #     count = 0
    #     for i in range(m):
    #         for j in range(n):
    #             if grid[i][j] == '1':
    #                 count += 1
    #                 dfs(i, j)
    #     return count
# leetcode submit region end(Prohibit modification and deletion)
