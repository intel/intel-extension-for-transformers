from functools import wraps
import time
import inspect

class TracerNode:
    def __init__(self, id, func_name, next=[]):
        # next: a list of TracerNode
        self.id = id
        self.func_name = func_name
        self.next = next

    def set_next(self, next_node):
        self.next.append(next_node)

    def set_time(self, time):
        self.time = time

class Tracer:

    def __init__(self):
        self.frame_to_tracer_node = {} # a mapping from the hashed frame to TracerNode
        self.root = None

    def _prepare_frame_tracer_nodes(self, func):
        cur_func_name = func.__name__
        cur_tracer_node = TracerNode(hash(func), cur_func_name, next=[])
        is_root = False
        if self.frame_to_tracer_node == {}:
            is_root = True
            self.root = cur_tracer_node

        if self.frame_to_tracer_node != {}:
            for i in inspect.stack()[1:]:
                # for every parent (not including self) recursively, check whether has a TracerNode
                # if there is a TracerNode, set its next to the current TracerNode, and break
                # else, continue checking until the run_endpoint
                # notice the frame should back to the wrapper which is on the stack of the callee func
                if hash(i.frame.f_back) in self.frame_to_tracer_node:
                    self.frame_to_tracer_node[hash(i.frame.f_back)].set_next(cur_tracer_node)
                    # print("\n")
                    # print(self.frame_to_tracer_node[hash(i.frame.f_back)].func_name)
                    # print("=>")
                    # print(self.frame_to_tracer_node[hash(i.frame.f_back)].next[0].func_name)
                    # print("\n")
                    break
        # append the currentframe of wrapper => current TracerNode to the dict
        cur_frame = hash(inspect.currentframe().f_back)

        self.frame_to_tracer_node[cur_frame] = cur_tracer_node
        return cur_tracer_node, is_root

    def time_tracer(self, func):
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                cur_tracer_node, is_root = self._prepare_frame_tracer_nodes(func)
                start = time.time()
                res = await func(*args, **kwargs)
                # print unstructured time
                # print(f'call {func.__name__}: {round(time.time() - start, 2)} seconds')
                cur_tracer_node.set_time(round(time.time() - start, 2))
                if is_root:
                    self.print_time_trace(self.root)
                    self.clearup()
                return res
        else:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cur_tracer_node, is_root = self._prepare_frame_tracer_nodes(func)
                start = time.time()
                res = func(*args, **kwargs)
                # print unstructured time
                # print(f'call {func.__name__}: {round(time.time() - start, 2)} seconds')
                cur_tracer_node.set_time(round(time.time() - start, 2))
                if is_root:
                    self.print_time_trace(self.root)
                    self.clearup()
                return res

        return wrapper

    def print_time_trace(self, cur_node, depth=0):
        if depth==0:
            prefix = ""
        else:
            prefix = "|" + "-"*depth*5 + "> "
        print(f'{prefix}{cur_node.func_name}: {cur_node.time} seconds')
        for next_node in cur_node.next:
            self.print_time_trace(next_node, depth+1)

    def clearup(self):
        self.frame_to_tracer_node = {}



tracer = Tracer()
