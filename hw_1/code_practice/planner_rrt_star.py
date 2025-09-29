
import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerRRTStar(Planner):
    def __init__(self, m, extend_len=20, neighbor_radius=50):
        """RRT* planner

        Args:
            m (np.ndarray): The map (2D array), 1 = free, 0 = obstacle
            extend_len (float): The distance to steer per iteration
            neighbor_radius (float): Radius used in 're-wire' step for local neighbors
        """
        super().__init__(m)  # runs Planner's __init__
        self.extend_len = extend_len
        self.neighbor_radius = neighbor_radius #全局參數

    def _random_node(self, goal, shape):
        """Randomly sample a node in free space
           (50% chance it directly samples the goal)
        """
        r = np.random.choice(2, 1, p=[0.5, 0.5]) #給與2,1 選擇并且持有個50%幾率
        if r == 1:
            # 50% chance: return the goal
            return (float(goal[0]), float(goal[1]))
        else:
            # 50% chance: random within map boundaries
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        """Find the nearest node in self.ntree to the sample node"""
        min_dist = 99999#不知道隨機點與初始有多大所以把min 調無限大
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        """Check if the line between n1, n2 collides with obstacles
           using Bresenham or any other ray-casting approach
        """
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            # If any pixel along the line is below 0.5, treat as collision
            if self.map[int(pts[1]), int(pts[0])] < 0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        """Generate a new node from 'from_node' toward 'to_node' 
           by 'extend_len', or less if the distance is smaller
        """
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])

        # If the distance is less than extend_len, go straight to to_node
        if extend_len > v_len:
            extend_len = v_len

        new_node = (from_node[0] + extend_len * np.cos(v_theta),
                    from_node[1] + extend_len * np.sin(v_theta))

        # Boundary check and collision check
        if (new_node[1] < 0 or new_node[1]>=self.map.shape[0] or new_node[0]<0 or new_node[0]>=self.map.shape[1] or
                    self._check_collision(from_node, new_node)):
            # If out of map or collision, return failure
            return False, None
        else:
            return new_node, utils.distance(new_node, from_node)

    def _find_neighbors(self, new_node):
        """Find all existing nodes within 'neighbor_radius' of new_node"""
        neighbors = []
        for node in self.ntree: #走遍所有 trees 裏的node把在定義半徑内的 node 都存在金neighbor
            dist = utils.distance(node, new_node)
            if dist < self.neighbor_radius:
                neighbors.append(node)
        return neighbors

    def planning(self, start, goal, extend_len=None, img=None):
        """Execute the RRT* planning routine

        Args:
            start (tuple): (x, y) start coordinate
            goal (tuple): (x, y) goal coordinate
            extend_len (float): override the default extension length
            img (np.ndarray): if provided, draw the progress on this image
        """
        if extend_len is None:
            extend_len = self.extend_len

        # Data structures for RRT*
        self.ntree = {}   # child -> parent
        self.ntree[start] = None
        self.cost = {}    # cost to reach each node from the start
        self.cost[start] = 0

        goal_node = None

        # RRT* iteration
        for it in range(20000):
            # 1. Sample a random node
            samp_node = self._random_node(goal, self.map.shape)

            # 2. Find the nearest node in the existing tree
            near_node = self._nearest_node(samp_node)

            # 3. Attempt to steer from near_node towards samp_node
            new_node, dist_cost = self._steer(near_node, samp_node, extend_len)
            if new_node is False:
                # Steering failed (collision or out of bounds), skip
                continue

            # 4. Temporarily connect new_node to near_node
            #    (We'll look for a better parent below)
            self.ntree[new_node] = near_node #把臨近與new nod 連一起
            self.cost[new_node] = self.cost[near_node] + dist_cost #from - 》to node 的距離

            # 5. RRT* "Re-Parent" step
            #    Find all nearby nodes and see which one offers the lowest cost
            neighbors = self._find_neighbors(new_node)
            best_parent = near_node
            best_cost = self.cost[new_node]

            for nb in neighbors:
                # Cost if we re-parent new_node to nb
                dist_nb_to_new = utils.distance(nb, new_node)
                new_cost_via_nb = self.cost[nb] + dist_nb_to_new #各個 nb 都有自己儲存的cost 把他們加上他們到 new node 的 距離 就是 new cost
                # Check collision if we connect nb -> new_node
                if (new_cost_via_nb < best_cost and 
                    not self._check_collision(nb, new_node)):
                    best_parent = nb #如果某nb之前的路經距離到new node 的距離加起來比原先只考慮隨機點最近的該點自身從 start 到 該點再到 newnode 的cost 還要小，該路徑就會直接把 一開始定義的 near node 替換。
                    best_cost = new_cost_via_nb #cost 也會重新計算  再遍歷所有圓形範圍内的 nb。

            # If we found a better parent than near_node, update #如果best parent 已不再是near node 就直接更新newnode 的關係 剔除 near_node
            if best_parent != near_node:
                self.ntree[new_node] = best_parent
                self.cost[new_node] = best_cost

            # 6. RRT* "Re-Wire" step
            #    For each neighbor, check if going through new_node improves cost 考慮所有的的 nb 自身的cost 從start 到nb 的， 把他們連到new node 再重新計算一次總cost。 
            for nb in neighbors:
                # Current cost to nb
                old_cost = self.cost[nb]
                # Cost if we rewire nb to go via new_node
                dist_nb_new = utils.distance(nb, new_node)
                new_cost_via_new = self.cost[new_node] + dist_nb_new #從start to new node 的距離再到 nb 如果有比之前 nb 的 距短就替換

                # If improved and no collision
                if (new_cost_via_new < old_cost and
                    not self._check_collision(new_node, nb)):
                    # Re-wire nb to new_node
                    self.ntree[nb] = new_node
                    self.cost[nb] = new_cost_via_new

            # 7. Goal check
            if utils.distance(new_node, goal) < extend_len:
                goal_node = new_node
                break

            # Draw
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0,1,0), 1)
                # Near Node
                img_ = img.copy()
                cv2.circle(img_,utils.pos_int(new_node),5,(0,0.5,1),3)
                # Draw Image
                img_ = cv2.flip(img_,0)
                cv2.imshow("Path Planning",img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        

        # =====================
        # Extract Path
        # =====================
        path = []
        n = goal_node
        while(True):
            if n is None:
                break
            path.insert(0,n)
            node = self.ntree[n]
            n = self.ntree[n] 
        path.append(goal)
        return path


# import cv2
# import numpy as np
# import sys
# sys.path.append("..")
# import PathPlanning.utils as utils
# from PathPlanning.planner import Planner


# import cv2
# import numpy as np
# import PathPlanning.utils as utils

# def draw_informed_ellipse(img, start, goal, c_best, color=(0,0,1), thickness=2):
#     """
#     Draw the ellipse for Informed RRT* sampling once we have a feasible path cost.
    
#     Args:
#         img (np.ndarray): The image where we draw the ellipse (CV_8UC3 or float).
#         start (tuple): (x, y) start coordinate.
#         goal (tuple): (x, y) goal coordinate.
#         c_best (float): The best (lowest) path cost found so far.
#         color (tuple): BGR color for drawing in OpenCV coordinate space (default: (0,0,1) = red if float).
#         thickness (int): Thickness of the drawn ellipse boundary.
#     """
#     # If we have not found any path yet, or c_best is invalid, skip
#     if np.isinf(c_best):
#         return

#     # Distance between start and goal
#     c_min = utils.distance(start, goal)

#     # If c_best < c_min (theoretically shouldn't happen after a path is found),
#     # or c_best is too small, skip
#     if c_best < c_min or c_min < 1e-9:
#         return

#     # Half of major axis
#     a = c_best / 2.0
#     # Half of minor axis
#     b = np.sqrt(c_best**2 - c_min**2) / 2.0
#     if b < 1e-9:
#         b = 1e-9  # avoid zero minor axis

#     # The midpoint between start and goal
#     mx = (start[0] + goal[0]) / 2.0
#     my = (start[1] + goal[1]) / 2.0

#     # Angle from start to goal
#     angle = np.arctan2(goal[1] - start[1], goal[0] - start[0])

#     # Generate a set of angle points to approximate the ellipse boundary
#     N = 200  # number of points on ellipse
#     ellipse_pts = []
#     for i in range(N+1):
#         t = 2.0 * np.pi * i / N
#         # (x_ellipse, y_ellipse) in ellipse local coords
#         x_ellipse = a * np.cos(t)
#         y_ellipse = b * np.sin(t)

#         # Rotate + translate into map frame
#         # [x']   =  [ cos(angle) -sin(angle)] [x_ellipse]
#         # [y']      [ sin(angle)  cos(angle)] [y_ellipse]
#         x_global = x_ellipse*np.cos(angle) - y_ellipse*np.sin(angle) + mx
#         y_global = x_ellipse*np.sin(angle) + y_ellipse*np.cos(angle) + my

#         ellipse_pts.append((int(x_global), int(y_global)))

#     # Draw as a closed polyline on the image
#     ellipse_pts_array = np.array(ellipse_pts, dtype=np.int32)
#     cv2.polylines(img, [ellipse_pts_array], isClosed=True, color=color, thickness=thickness)

# class PlannerRRTStar(Planner):
#     def __init__(self, m, extend_len=20, neighbor_radius=50):
#         """Informed RRT* planner

#         Args:
#             m (np.ndarray): The map (2D array), 1 = free, 0 = obstacle
#             extend_len (float): The distance to steer per iteration
#             neighbor_radius (float): Re-wire radius
#         """
#         super().__init__(m)
#         self.extend_len = extend_len
#         self.neighbor_radius = neighbor_radius

#         # Data structures
#         self.ntree = {}   # child -> parent
#         self.cost = {}    # cost to reach each node from the start
#         self.best_cost = float('inf')  # The current best path cost
#         self.goal_node = None

#     def _check_collision(self, n1, n2):
#         """Check if the line between n1, n2 collides with obstacles
#            using Bresenham or any other ray-casting approach.
#         """
#         n1_ = utils.pos_int(n1)
#         n2_ = utils.pos_int(n2)
#         line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
#         for pts in line:
#             if self.map[int(pts[1]), int(pts[0])] < 0.5:
#                 return True
#         return False

#     def _nearest_node(self, samp_node):
#         """Find the nearest node in self.ntree to the sample node."""
#         min_dist = float('inf')
#         min_node = None
#         for n in self.ntree:
#             dist = utils.distance(n, samp_node)
#             if dist < min_dist:
#                 min_dist = dist
#                 min_node = n
#         return min_node

#     def _steer(self, from_node, to_node, extend_len):
#         """Generate a new node from 'from_node' toward 'to_node' 
#            by 'extend_len', or less if the distance is smaller.
#         """
#         vect = np.array(to_node) - np.array(from_node)
#         v_len = np.hypot(*vect)
#         if v_len == 0:
#             return False, None

#         v_theta = np.arctan2(vect[1], vect[0])
#         if extend_len > v_len:
#             extend_len = v_len

#         new_node = (from_node[0] + extend_len * np.cos(v_theta),
#                     from_node[1] + extend_len * np.sin(v_theta))

#         # Boundary & collision checks
#         if (new_node[1] < 0 or new_node[1] >= self.map.shape[0] or
#             new_node[0] < 0 or new_node[0] >= self.map.shape[1] or
#             self._check_collision(from_node, new_node)):
#             return False, None
        
#         dist_cost = utils.distance(from_node, new_node)
#         return new_node, dist_cost

#     def _find_neighbors(self, new_node):
#         """Return all existing nodes within 'neighbor_radius' of new_node."""
#         neighbors = []
#         for node in self.ntree:
#             if utils.distance(node, new_node) < self.neighbor_radius:
#                 neighbors.append(node)
#         return neighbors

#     def _random_node_informed(self, start, goal):
#         """
#         Generate a random node using 'informed sampling' if we have
#         a known best path cost. If no best path is found yet, sample
#         from the entire map.
#         """
#         # If no feasible path yet, sample the entire space
#         if self.best_cost == float('inf'):
#             return (np.random.uniform(0, self.map.shape[1]),
#                     np.random.uniform(0, self.map.shape[0]))

#         # cBest = current best cost
#         cBest = self.best_cost
#         # cMin = distance between start and goal
#         cMin = utils.distance(start, goal)
#         if cMin == 0:
#             # Edge case: start == goal
#             return (float(goal[0]), float(goal[1]))

#         # If cBest < cMin (shouldn't happen logically if a path is found),
#         # just sample full space
#         if cBest < cMin:
#             return (np.random.uniform(0, self.map.shape[1]),
#                     np.random.uniform(0, self.map.shape[0]))

#         # Compute ellipse axes:
#         # major axis = cBest
#         # minor axis = sqrt(cBest^2 - cMin^2)
#         a = cBest / 2.0
#         b = np.sqrt(cBest**2 - cMin**2) / 2.0
#         if b < 1e-9:
#             b = 1e-9  # Avoid zero minor axis

#         # Sample a random point from the unit circle (u, v)
#         # in elliptical coordinates: x = a * u, y = b * v
#         # where (u^2 + v^2 <= 1)
#         theta = np.random.uniform(0, 2*np.pi)
#         r = np.sqrt(np.random.uniform(0, 1))
#         u = r * np.cos(theta)
#         v = r * np.sin(theta)
#         # Scale to ellipse space
#         x_ellipse = a * u
#         y_ellipse = b * v

#         # Now we must rotate+translate the point so that the ellipse
#         # is aligned with the line from start to goal.
#         # 1) Build the rotation from start->goal angle
#         # 2) Translate so that ellipse is “centered” at the midpoint
#         #    between start and goal.

#         # The midpoint between start and goal
#         mx = (start[0] + goal[0]) / 2.0
#         my = (start[1] + goal[1]) / 2.0

#         # Angle from start to goal
#         angle = np.arctan2(goal[1] - start[1], goal[0] - start[0])

#         # Rotation matrix
#         rot = np.array([
#             [np.cos(angle), -np.sin(angle)],
#             [np.sin(angle),  np.cos(angle)]
#         ])

#         # Rotate the point in ellipse coords
#         pt_ellipse = np.dot(rot, np.array([x_ellipse, y_ellipse]))
#         # Translate to map coords
#         rx = pt_ellipse[0] + mx
#         ry = pt_ellipse[1] + my

#         # Clip to map boundaries just in case
#         rx = np.clip(rx, 0, self.map.shape[1]-1)
#         ry = np.clip(ry, 0, self.map.shape[0]-1)
#         return (float(rx), float(ry))

#     def planning(self, start, goal, extend_len=None, img=None):
#         if extend_len is None:
#             extend_len = self.extend_len

#         # Initialize tree and cost
#         self.ntree = {}
#         self.ntree[start] = None
#         self.cost = {}
#         self.cost[start] = 0
#         self.best_cost = float('inf')
#         self.goal_node = None

#         # Precompute direct distance between start and goal
#         direct_dist = utils.distance(start, goal)

#         max_iter = 20000
#         for it in range(max_iter):
#             # ============= Informed Sampling =============
#             samp_node = self._random_node_informed(start, goal)

#             # Standard RRT steps
#             near_node = self._nearest_node(samp_node)
#             new_node, dist_cost = self._steer(near_node, samp_node, extend_len)
#             if new_node is False:
#                 continue

#             # Add new_node to tree with near_node as parent
#             self.ntree[new_node] = near_node
#             self.cost[new_node] = self.cost[near_node] + dist_cost

#             # =================== Re-Parent (RRT*) ===================
#             neighbors = self._find_neighbors(new_node)
#             best_parent = near_node
#             best_cost = self.cost[new_node]

#             for nb in neighbors:
#                 dist_nb_new = utils.distance(nb, new_node)
#                 candidate_cost = self.cost[nb] + dist_nb_new
#                 if (candidate_cost < best_cost and
#                     not self._check_collision(nb, new_node)):
#                     best_parent = nb
#                     best_cost = candidate_cost

#             if best_parent != near_node:
#                 # Update parent to best_parent
#                 self.ntree[new_node] = best_parent
#                 self.cost[new_node] = best_cost

#             # ==================== Re-Wire (RRT*) ====================
#             for nb in neighbors:
#                 dist_nb_new = utils.distance(nb, new_node)
#                 candidate_cost = self.cost[new_node] + dist_nb_new
#                 if (candidate_cost < self.cost[nb] and
#                     not self._check_collision(new_node, nb)):
#                     # Rewire neighbor
#                     self.ntree[nb] = new_node
#                     self.cost[nb] = candidate_cost

#             # =================== Goal Check ===================
#             if utils.distance(new_node, goal) < extend_len:
#                 # Found a path to the goal region
#                 # Update best cost if this path is better
#                 new_total_cost = self.cost[new_node] + utils.distance(new_node, goal)
#                 if new_total_cost < self.best_cost:
#                     self.best_cost = new_total_cost
#                     self.goal_node = new_node

#             if img is not None and it % 50 == 0:
#                 # Draw the RRT* tree edges
#                 for n in self.ntree:
#                     p = self.ntree[n]
#                     if p is None:
#                         continue
#                     cv2.line(img, utils.pos_int(n), utils.pos_int(p), (0,1,0), 1)
                
#                 # Draw the ellipse for the informed sampling region
#                 draw_informed_ellipse(img, start, goal, self.best_cost, color=(0,0,1), thickness=2)

#                 # Draw the newest node in a highlight color
#                 cv2.circle(img, utils.pos_int(new_node), 3, (0,0.5,1), -1)

#                 # Show
#                 img_ = cv2.flip(img, 0)  # Flip vertically if needed
#                 cv2.imshow("Informed RRT* with Ellipse", img_)
#                 k = cv2.waitKey(1)
#                 if k == 27:
#                     break


#             # If we have a feasible path, we try to keep improving until out of iterations
#             # or until the improvement is minimal. Here we just run up to max_iter.

#         # ============== Path Extraction ==============
#         path = []
#         if self.goal_node is None:
#             # No path found
#             return path

#         # Build path by backtracking from self.goal_node
#         n = self.goal_node
#         while n is not None:
#             path.insert(0, n)
#             n = self.ntree[n]

#         # Optionally ensure exact goal is appended
#         if utils.distance(path[-1], goal) > 1:
#             path.append(goal)

#         return path
