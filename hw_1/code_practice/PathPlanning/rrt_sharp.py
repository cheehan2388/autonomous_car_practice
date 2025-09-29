import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner
from collections import deque

class PlannerRRTSharp(Planner):
    def __init__(self, m, extend_len=20, neighbor_radius=50, max_iter=2000):
        """
        RRT# (RRT Sharp) 路径规划器
        
        参数:
            m (np.ndarray): 地图（二维数组），1=空闲，0=障碍
            extend_len (float): 扩展步长
            neighbor_radius (float): 邻域半径，用于选择更优父节点以及重新布线
            max_iter (int): 最大迭代次数
        """
        super().__init__(m)   # 调用父类的初始化
        self.m = m           # 地图
        self.extend_len = extend_len
        self.neighbor_radius = neighbor_radius
        self.max_iter = max_iter

        # 主要数据结构
        self.ntree = {}          # child -> parent
        self.cost = {}           # 记录从起点到每个节点的累计代价
        self.children = {}       # parent -> 子节点集合，便于做代价更新传播
        self.inconsistency_queue = deque()  # 用于存储等待更新的节点

        self.start = None
        self.goal = None

    def _random_node(self, goal, shape):
        """
        随机生成一个节点或直接返回目标节点（可自定义概率）。
        这里示例用 10% 概率采样目标，提高扩展向目标的概率。
        """
        if np.random.rand() < 0.1:
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(shape[1]))
            ry = float(np.random.randint(shape[0]))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        """
        在当前树中找离 samp_node 空间距离最近的节点。
        """
        min_dist = float('inf')
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        """
        碰撞检测：
        使用 Bresenham 或其他方法，检查从 n1 到 n2 的线段上是否经过障碍物。
        """
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pt in line:
            if (pt[0] < 0 or pt[0] >= self.m.shape[1] or
                pt[1] < 0 or pt[1] >= self.m.shape[0]):
                return True  # 越界
            if self.m[int(pt[1]), int(pt[0])] < 0.5:  # 认为 <0.5 是障碍
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        """
        从 from_node 向 to_node 方向扩展距离 extend_len，
        如果距离小于 extend_len，直接到达 to_node。
        返回 (new_node, dist_cost)。
        若碰撞或越界，返回 (False, None)。
        """
        vect = np.array(to_node) - np.array(from_node)
        dist = np.hypot(vect[0], vect[1])
        if dist < 1e-9:  # 防止除0
            return False, None

        if dist > extend_len:
            ratio = extend_len / dist
            new_node = (from_node[0] + ratio * vect[0],
                        from_node[1] + ratio * vect[1])
        else:
            new_node = to_node

        # 边界或碰撞检测
        # 注意：这里仅在返回后再做碰撞检测也可以，
        # 也可在这里先行判断下是否越界
        if (new_node[0] < 0 or new_node[0] >= self.m.shape[1] or
            new_node[1] < 0 or new_node[1] >= self.m.shape[0]):
            return False, None

        dist_cost = dist if dist < extend_len else extend_len
        return new_node, dist_cost

    def _find_neighbors(self, node):
        """
        在邻域半径内搜索节点，用于重新选择父节点和重新布线。
        """
        neighbors = []
        for nd in self.ntree:
            if utils.distance(nd, node) < self.neighbor_radius:
                neighbors.append(nd)
        return neighbors

    def _init_tree(self, start):
        """
        初始化树：清空所有结构，并将起点加到树中。
        """
        self.ntree.clear()
        self.cost.clear()
        self.children.clear()
        self.inconsistency_queue.clear()

        self.ntree[start] = None  # 起点没有父节点
        self.cost[start] = 0      # 起点的代价为0
        self.children[start] = set()

    def _insert_node(self, node, parent, total_cost):
        """
        将新节点插入树中，并建立父子关系。
        """
        self.ntree[node] = parent
        self.cost[node] = total_cost
        self.children[node] = set()
        if parent is not None:
            self.children[parent].add(node)

    def _change_parent(self, node, new_parent, new_cost):
        """
        RRT# 专用函数：更改节点的父节点并更新 cost，适配 children 字典。
        同时将该 node 加入 inconsistency_queue。
        """
        old_parent = self.ntree[node]
        if old_parent is not None and node in self.children[old_parent]:
            self.children[old_parent].remove(node)

        self.ntree[node] = new_parent
        self.cost[node] = new_cost

        if new_parent is not None:
            self.children[new_parent].add(node)

        # 将 node 加入队列，以便后续向子节点传播代价改进
        self.inconsistency_queue.append(node)

    def _propagate_cost_improvements(self):
        """
        RRT# 特有流程：如果某个节点的代价改进了，就可能进一步改进其子节点的代价。
        因此用一个队列来逐个处理可能需要更新的节点，直到没有可改进的情况。
        """
        while self.inconsistency_queue:
            cur = self.inconsistency_queue.popleft()
            # 检查 cur 的所有子节点，看看是否可以通过 cur 的最新代价来改进
            for c in list(self.children[cur]):
                old_cost = self.cost[c]
                new_cost = self.cost[cur] + utils.distance(cur, c)
                if new_cost < old_cost:
                    # 如果可以改进，就更新 c 的父节点为 cur，代价更新
                    self._change_parent(c, cur, new_cost)

    def planning(self, start, goal, extend_len=None, img=None):
        """
        RRT# 主函数：
        1. 随机采样并生成新节点
        2. 重新选择父节点（Re-Parent）
        3. 重新布线（Re-Wire）
        4. 传播代价改进（_propagate_cost_improvements）
        """
        if extend_len is None:
            extend_len = self.extend_len

        self.start = start
        self.goal = goal

        # 初始化树
        self._init_tree(start)

        for it in range(self.max_iter):
            # 1) 随机采样
            samp_node = self._random_node(goal, self.m.shape)

            # 2) 找到最近节点
            near_node = self._nearest_node(samp_node)
            if near_node is None:
                continue

            # 3) 扩展
            new_node, dist_cost = self._steer(near_node, samp_node, extend_len)
            if new_node is False:
                continue

            # 4) 碰撞检测
            if self._check_collision(near_node, new_node):
                continue

            # 先把 new_node 接入到 near_node
            cost_new = self.cost[near_node] + dist_cost
            self._insert_node(new_node, near_node, cost_new)

            # ========== Re-Parent: 查找邻域中是否有更优的父节点 ==========
            neighbors = self._find_neighbors(new_node)
            best_parent = near_node
            best_cost = cost_new
            for nb in neighbors:
                if nb == new_node:
                    continue
                tmp_cost = self.cost[nb] + utils.distance(nb, new_node)
                if (tmp_cost < best_cost and
                    not self._check_collision(nb, new_node)):
                    best_parent = nb
                    best_cost = tmp_cost

            # 如果找到更优父节点，更新
            if best_parent != near_node:
                self._change_parent(new_node, best_parent, best_cost)

            # ========== Re-Wire: 利用 new_node 改进邻域节点的代价 ==========
            for nb in neighbors:
                if nb == new_node:
                    continue
                new_cost_via_new = self.cost[new_node] + utils.distance(new_node, nb)
                if (new_cost_via_new < self.cost[nb] and
                    not self._check_collision(new_node, nb)):
                    # 更新 nb 的父节点为 new_node
                    self._change_parent(nb, new_node, new_cost_via_new)

            # ========== RRT# 关键: 传播代价改进 ==========
            self._propagate_cost_improvements()

            # ========== 绘制（可选） ==========
            if img is not None and it % 50 == 0:
                # 画出当前树的边
                for node in self.ntree:
                    p = self.ntree[node]
                    if p is None:
                        continue
                    cv2.line(img, utils.pos_int(node), utils.pos_int(p), (0,1,0), 1)
                # 画新节点
                img_ = img.copy()
                cv2.circle(img_, utils.pos_int(new_node), 5, (0,0.5,1), 3)
                img_ = cv2.flip(img_, 0)
                cv2.imshow("RRT#", img_)
                k = cv2.waitKey(1)
                if k == 27:  # ESC
                    break

            # 可加速收敛：如果已接近目标，就提前结束
            if utils.distance(new_node, goal) < extend_len:
                # 再做一次改进传播
                self._propagate_cost_improvements()
                break

        # 提取路径
        path = self._extract_path(goal)
        return path

    def _extract_path(self, goal):
        """
        从最接近 goal 的节点开始回溯到起点，生成路径。
        如果想要精确目标，可再补充一个将终点插入的逻辑。
        """
        # 先找树中最接近 goal 的节点
        closest_node = None
        min_dist = float('inf')
        for node in self.ntree:
            dist_ = utils.distance(node, goal)
            if dist_ < min_dist:
                min_dist = dist_
                closest_node = node

        if closest_node is None:
            return []

        # 逆向回溯
        path = []
        cur = closest_node
        while cur is not None:
            path.insert(0, cur)
            cur = self.ntree[cur]

        # 如果要保证把真正的 goal 点也纳入
        # 可以在 path 最后追加
        if utils.distance(path[-1], goal) > 1.0:
            path.append(goal)

        return path
