import numpy as np
import cv2
from collections import deque

white = [255, 255, 255]
maze2 = cv2.imread("chennaimap.png", 0)
maze = cv2.imread("chennaimap.png", 1)
org = maze.copy()
black = [0, 0, 0]

l, w, t = maze.shape
for i in range(l):
    for j in range(w):
        for k in range(1, 3):
            if maze[i][j][k] >= 240:
                maze[i][j][k] = 255
                if k == 2:
                    maze[i][j] = white

            else:
                maze[i][j][k] = 0
        if maze[i][j][0] == 255 and maze[i][j][1] == 255 and maze[i][j][2] == 255:
            pass
        else:
            maze[i][j] = black

        if maze2[i][j] >= 250 or (maze2[i][j] >= 10 and maze2[i][j] <= 230):
            maze2[i][j] = 255
        else:
            maze2[i][j] = 0
            maze[i][j] = black

cv2.imshow("map", maze)
cv2.imshow("maporg", org)
cv2.waitKey(0)

white = [255, 255, 255]
green = [0, 255, 0]
black = [0, 0, 0]
gray = [127, 127, 127]

a = 100

start = (310, 71)
# end=(330,474)
end = (340, 162)
def dijkstraspath():
    img = np.array(maze)
    l, w, t = img.shape

    def calcDist(point, current):
        return ((point[0] - current[0]) ** 2 + (point[1] - current[1]) ** 2) ** (1 / 2)

    def iswell(img, x, y):
        return (x >= 0 and x < img.shape[0] and y >= 0 and y < img.shape[1])

    def dijkstra(img, start, end):
        h, w, t = img.shape

        dist = np.full((h, w), fill_value=np.inf)
        dist[start] = 0
        parent = np.zeros((h, w, 2))
        visited = np.zeros((h, w))
        visited[start] = 1
        current = start
        while current != end:

            visited[current] = 1
            for i in range(-1, 2):
                for j in range(-1, 2):
                    point = (current[0] + i, current[1] + j)
                    if iswell(img, point[0], point[1]) and (
                            img[point][0] == white[0] and (img[point][1] == white[1] and img[point][2] == white[2]) or (
                            img[point][0] == gray[0] and img[point][1] == gray[1] and img[point][2] == gray[2])):
                        if (calcDist(point, current) + dist[current] < dist[point]):
                            dist[point] = calcDist(point, current) + dist[current]
                            parent[point[0], point[1]] = [current[0], current[1]]

            min = np.inf
            for i in range(h):
                for j in range(w):
                    if min > dist[i, j] and visited[i, j] != 1:
                        min = dist[i, j]
                        current = (i, j)
            showPath(img, current, start, parent)

    def showPath(img, current, start, parent):
        new = np.copy(img)
        while current != start:
            var = int(parent[current][0]), int(parent[current][1])

            new[int(var[0]), int(var[1])] = green
            current = (var[0], var[1])

        cv2.namedWindow('dijkstras path', cv2.WINDOW_NORMAL)
        cv2.imshow('dijkstras path', new.astype(np.uint8))
        cv2.waitKey(1)

    dijkstra(img, start, end)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def astarpath():
    global maze
    global start
    global end
    img = np.array(maze)
    startastar = (start[1], start[0])
    endastar = (end[1], end[0])

    img_copy = np.array(maze)

    green = (0, 255, 0)

    orange = (0, 128, 255)
    pink = (255, 0, 255)

    w, h, c = img.shape

    class Node():
        def __init__(self, parent, position):
            self.parent = parent
            self.position = position
            self.g = np.inf
            self.h = np.inf
            self.f = np.inf

    def get_min_dist_node(open_list):
        min_dist = np.inf
        min_node = None
        for node in open_list:
            if open_list[node].f < min_dist:
                min_dist = open_list[node].f
                min_node = open_list[node]
        return min_node

    def show_path(end, start, img):

        current = end.parent

        while (current.position != start.position):
            x = current.position[1]
            y = current.position[0]
            img[x][y] = green
            current = current.parent
        cv2.namedWindow('final path', cv2.WINDOW_NORMAL)
        cv2.imshow("final path", img.astype(np.uint8))

    def get_dist(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return (((x1 - x2) ** 2 + (y1 - y2) ** 2)) ** 0.5

    def obstacle(position):
        x, y = position
        if x==396 or y==396:
            return True
        if y!=396 and x!=396 and img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 0  :
            return True
        return False

    def goal_reached(position):
        x, y = position
        if (x, y) == end:
            return True
        return False

    def astar_algorithm(start, end):

        open_list = {}
        closed_list = []
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        open_list[start] = start_node
        while len(open_list) > 0:

            current_node = get_min_dist_node(open_list)
            img[current_node.position[1]][current_node.position[0]] = orange
            open_list.pop(current_node.position)

            if current_node.position == end:
                print("Goal Reached")
                endnode = Node(current_node.parent, endastar)
                startnode = Node(None, startastar)
                show_path(endnode, startnode, img_copy)
                return

            for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
                if node_position[0] > (w - 1) or node_position[0] < 0 or node_position[1] > (h - 1) or node_position[
                    1] < 0:
                    continue
                if node_position in closed_list:
                    continue
                if obstacle(node_position):
                    continue

                img[node_position[1]][node_position[0]] = pink
                new_node = Node(current_node, node_position)

                new_node.g = current_node.g + get_dist(current_node.position, new_node.position)
                new_node.h = get_dist(new_node.position, end)
                new_node.f = new_node.g + new_node.h

                if new_node.position in open_list:
                    if new_node.g < open_list[new_node.position].g:
                        open_list[new_node.position] = new_node
                else:
                    open_list[new_node.position] = new_node

            if current_node.position not in closed_list:
                closed_list.append(current_node.position)

            cv2.namedWindow('path_finding', cv2.WINDOW_NORMAL)
            cv2.imshow("path_finding", img.astype(np.uint8))
            cv2.waitKey(1)

    if __name__ == '__main__':
        astar_algorithm(startastar, endastar)

        cv2.namedWindow("path_finding", cv2.WINDOW_NORMAL)
        cv2.imshow("path_finding", img.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

astarpath()
dijkstraspath()