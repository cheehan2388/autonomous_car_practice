import cv2
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerAStar(Planner): #創造class 繼承 planner 父
    def __init__(self, m, inter=10):
        self.m = m 
        super().__init__(m)
        self.inter = inter #步數如果10 就是 每10 pixel一個方方.
        self.initialize()

    def initialize(self):
        self.queue = []
        self.parent = {}
        self.h = {} # Distance from start to node heuristic
        self.g = {} # Distance from node to goal 會存 (y,x)： distance
        self.goal_node = None

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:#Uses the default inter if none is provided
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize 
        self.initialize()
        self.queue.append(start)
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal) #格子終點最短距離
        moves = []
        #初始化從原點延申各方向的路
        for dx in [-inter, 0, inter]:
          for dy in [-inter, 0, inter]:
            if dx == 0 and dy == 0:
                continue
            moves.append((dx, dy))
        
        while(1):

        # If queue is empty, no path exists
            if not self.queue:
                break

            # 1. Pick the node with the smallest f = g[node] + h[node] （開始只能選start
            current = min(self.queue, key=lambda n: self.g[n] + self.h[n])#min(矩陣用來找裏面最小值，回傳queue 裏滿足 g+h 最小的數字對)
            # 2. Check if we've reached the goal (or are sufficiently close)
            if current == goal:
                self.goal_node = current
                break

            # Remove current from the open set (queue,and use the new current)
            self.queue.remove(current) #選過了移除。
           
            # 3. Check all possible neighbors（現在查點附近的五鄰右）
            for dx, dy in moves: 
                nx = current[0] + dx #let cur at the moves define above (inter)
                ny = current[1] + dy

                # Obstacle check
                if self.m[ny, nx] <= 0:#障礙已經被設定=0 了
                    continue

                # Cost from current to neighbor 之前的cost 加上之前的點到下一個點的 cos 就是 g
                new_cost = self.g[current] + utils.distance(current, (nx, ny))

                # 4. 確定x,y 沒被存過字典或者 他的值比之前還要小
                if (nx, ny) not in self.g or new_cost < self.g[(nx, ny)]: #[(nx,ny)]這裏在拿鍵值
                    self.g[(nx, ny)] = new_cost
                    self.parent[(nx, ny)] = current
                    self.h[(nx, ny)] = utils.distance((nx, ny), goal)
                    if (nx, ny) not in self.queue:
                        self.queue.append((nx, ny))#更新 queue


        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while(True):
            path.insert(0,p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
