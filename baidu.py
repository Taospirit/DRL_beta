class yimian1():
    def __init__(self):
        pass
    def get_index(self, num_list, target):
        l, r = 0, len(num_list) - 1
        while l != r:
            if num_list[l] + num_list[r] > target:
                r -= 1
            if num_list[l] + num_list[r] < target:
                l += 1
            if num_list[l] + num_list[r] == target:
                return l, r
        return 0, 0

class yimian2():
    def __init__(self):
        pass
    def get_num(self, num_list):
        a = 0
        for item in num_list:
            a ^= item
        return a

class ermian():
    def __init__(self):
        self.choose = []
        self.get = False
    def backsum(self, num_list, m, target):
        if self.get or len(self.choose) > m:
            return
        if sum(self.choose) == target and len(self.choose) == m:
            print (self.choose)
            self.get = True
            return
        for i in range(len(num_list)):
            self.choose.append(num_list[i])
            self.backsum(num_list[:i] + num_list[i+1:], m, target)
            self.choose.pop()