# import re
#
# def find_all_indexes(sub,s):
#     # 使用正则表达式查找所有子字符串sub在字符串s中的位置
#     pattern = re.compile(re.escape(sub))
#     matches = pattern.finditer(s)
#     return [match.start() for match in matches]
#
#
# s = "Alibaba reports strong Q2 growth while Alibaba faces regulatory challenges."
# sub = "Alibaba"
# indexes = find_all_indexes(sub,s)
# print(indexes)


def find_all_indexes(s, sub):
    indexes = []
    i = 0
    while True:
        i = s.find(sub, i)
        if i == -1:
            return indexes
        indexes.append(i)
        i += 1  # 移动到下一个字符，避免无限循环
    return indexes


# s = "这里有一些文本，文本很重要，我们需要找到所有的文本。"
# sub = "文本"
# indexes = find_all_indexes(s, sub)
# print(indexes)


try:
    # 尝试执行的代码块
    li = []
    print(li[0])
    pass
except Exception as e:
    # 如果发生指定类型的异常，则执行这里的代码
    print(f"发生了一个错误：{e}")
    pass
else:
    # 如果try块中没有发生异常，则执行这里的代码
    pass
finally:
    # 无论是否发生异常，都会执行这里的代码
    pass