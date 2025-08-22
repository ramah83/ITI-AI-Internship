# data structure in python

# List , Tuple , Dictionary , set
# list - mutable, any kind of data or data type

# bracket []-> [0,1,2,3,..]
# seperate between item using comma ','

# lst = [1,2,3,4]
# print(len(lst)) # -> 4
#
# # check num in lst
# print(4 in lst) # condition -> True or false
#
# print list container
# for i in lst:
#     print(i, end = " ")
# print()
# print(lst)

# List can contain differnt data types
#  index of list start from 0
# lst = [1, "iti",5.3]
# print(len(lst))
# print(lst[1]) # iti
# lst[1] = 32
# print(lst[1]) 
# print(lst)
# index out of range -> range(2)
# lst[2] = 0
# print(lst)
# print(lst[3])
# for i in lst:
#     print(i)

# build list from scratch
# create empty list
# lst = list()
# print(lst) #-> [] , list()
# i need to add 2 elem.
# append -> add
# lst.append("Book")
# lst.append(99)
# print(lst)
# lst.append('Cookies')
# print(lst)

# print(lst.pop())
# print(lst)
# print(lst[-1])

# lst = [5,3,4,8,9]
# 0  1  2  3  4 -> index of value
# 5  3  4  8  9 -> value
#-5 -4 -3 -2 -1

# lst = [5,3,4,8,9]
# print(lst[-3])

#
# Mutable
# lst = [1,2,3,4,5]
# lst[0] = 9
# print(lst)

# indexing
# num = [10 ,2, 7, 5, 3]
# num[0] = 9
# num[2] *=3
# num[4] +=1
# print(num[4])
# print(num)

# for idx in range(5): # 0 -> 4
#     print(num[idx] , end= ' ')

# index -> 0 1 2 3 4
# numbe -> 9 2 21 5 4

#  + and * operators

# lst = [1, 'ITI' , 5]
# another_list = [99, 11.5]
# # concatination lists
# conc_lst = lst + another_list
# print(conc_lst)
# #
# thrd_lst = 3 * conc_lst
# print(thrd_lst)

# Append , extend and insert
# lst = [1,2,3,4,5]
# lst.append('Hello')
# print(lst)
# #
# another_lst = [3,1]
# another_lst.extend(lst)
# print(another_lst)
# print(lst)

# #insert take position and value
# lst.insert(2 , "wow")
# print(lst)
# for i in lst:
#     print(i , end = " ")
# print()

# Pop , remove , del
#
# lst = [1,5,10,17, 2,"ITI"]
# print(lst.pop())
# print(lst)
# # take index
# # print(lst.pop(3))

# # remove take value
# print(lst.remove(17))
# print(lst)

# # del take index
# del lst[0]
# print(lst)

# # clear
# lst.clear()
# print(lst)

# lst = [1,2,3]
# print(id(lst))
# ________________________________________
# function
# bolck -> of code
# print("Hello")
# print("ITI")
# print("Python")

# def Printing():
#     print("Hello")
#     print("ITI")
#     print("Python")

# Printing()
# Printing()

#def-> define function
# def funName(parameter):

# def nums_op(fNum , sNum):
#     print("Sum: "+str(fNum + sNum))
#     print("Multiplication: "+str(fNum * sNum))

# nums_op(5,6)


"""
1 2 3 5 10 12 15 18 27 100 -> increase
1 2 3 5 10 6 12 15 -> not increasing 
"""



# slicing in list
# lst = [1,2,3,4,5,6,7,8,9,10]
# print(type(lst))

# print(type(range(5)))

# slice_lst = lst[2:8]
# print(slice_lst)
# [start:end-1:step]

# sl_lst = lst[2:] # from index 2 to end
# sl_lst = lst[2:100000]
# print(sl_lst)
# sl_lst = lst[:5]
# print(sl_lst)
# sam_val = lst[:4] + lst[4:6]
# print(sam_val)
# print(lst[:])

# slice with steps
# slicing list[st:end:stp]
# lst = [1,2,3,4,5,6,7,8,9,10]
# sub_lst = lst[: :-2]
# print(sub_lst)
# print(lst[::-1])

# list comperehinsion
# lst1 = [1,2,3,4,5,6,7,8,9,10]
# lst2 = []
# for i in lst1:
#     lst2.append(i*i+1)
# print(lst2)

# list comperehinsion
# lst2 = [expression for increment in lst]

# lst2 = [i*i+1 for i in lst1]
# print(lst2)
# lst = [1,2,-4,3,-3,-7,6,-1]
# mn = lst[0]
# for i in lst:
#     if(mn>i):
#         mn = i
# print(mn)

# print(min(lst))
# print(max(lst))
# print(sorted(lst))

# lst2 = []
# for i in lst:
#     if i > 0:
#         lst2.append(i)
# print(lst2)
# #
# lst3 = [i for i in lst if i>0]
# print(lst3)

# search for else and elif in list comperhension


# Tuples
# another an order collections of objects
# similer list
# iter , indexing, slicing, comparason, func -> min, max, sorted()

# more
"""
immutable data type: we can't change or delete it's item
many methods -> append , insert , remove
"""

# tup = (5,6,7)
# x,y,z = tup
# print(x,y,z)

"""
swap in one line
"""
# x,y = y,x
# print(x,y,z)

# lst =[1,2,3, [4,5,6,[7,8,9]], [10,11,12]]

# creation
# tup = ("iti" , 12, 3.4 , [1,2,3])
# print(tup)
# print(len(tup))
# print(type(tup))

# t = (10)
# print(type(t))
#
# t = (10,)
# print(type(t))

# t = tuple('ITI')
# print(t)
# print(type(t))
#
# t = tuple(1,2,3)
# print(t)
#
# t = tuple([[1,2,3],[4,5,6]])
# print(len(t))
# print(t)

# indixing and slicing
# num = (1,2,3,6,4,5,3,1,6,1,3,3,6)
# print(num[0] , num[5])
# print(num[1:4])
# print(num[::])
# # methods and function
# print(num.count(3))

# print(num.index(6)) #-> index of value

# print(min(num),'+',max(num))
# print(sorted(num))

# reversed
# num = (1,2,3,4,5,6)
# print(tuple(reversed(num))) # -> 6,5,4,3,2,1

# t1 = tuple([[1],[2],[3]])
# print(t1)
# print(len(t1))

#  + and * operators
# t1 = (1,2,3)
# print(type(t1))
# t2 = ("ITI", True)
# # #

# t = t1+ 2*t2
# print(t)
# print("hi "*2)

# tup = 1,2,3
# x,y,z = tup
# x = 1
# y = 2
# z = 3
# Unpacking
# lst = 1,2,3,4,5
# print(lst)
# print(type(lst))
# a,b,*c = lst
"""
a = 1:-> int
b = 2:-> int
*c = [3,4,5]:-> list
"""
# print(a, type(a))
# print(b, type(b))
# print(c, type(c))

# *a,b,c = lst
# print(a)
# print(b)
# print(c)

# a,*b,c = lst
# print(a)
# print(b)
# print(c)

"""
num = 1,2,3,4,5
a,b,c = num

unpacking
a,b, *c = num

a, *b ,c = num

*a , b,c = num
"""

# lst1 = [1,2,3]
# lst2 = [4,5,6]

# conc = [*lst1 , *lst2]
# print(conc)

"""
Dictionary and Set
dictionary like list but it have key, value
key -> immutable
value -> mutable
"""

# lst = [1,2,3]
# print(lst[1]) # -> index

# dictionary = {
#     Key : Value
#   }
# dic = {0:1 , 1:2 , 213: 3}
# print(dic[213])
# print(dic[0])

# dic = {
#     'p' : 'Python',
#     'i' : "ITI",
#     'w' : "Winter"
# }
# print(dic)
# print(dic.keys())
# print(dic.values())
# print(dic['p'])

# update and delete
# dic = {}

# key       #value
# dic[12] = [234,(1,"ITI")]
# print(dic)
# # dic = {12:[234,(1,"ITI")]}
# dic["ITI"] = 20
# # dic = {12:[234,(1,"ITI")], "ITI":20 }
# print(dic.keys())
# print(dic.values())
# print(dic)

# del func
# del dic[12]
# print(dic)
# print(type(dic))

# Indinxing dic values
# dic = {
#     "ITI": "Learn",
#     1: [1,5,7,8],
#     3: [[3,4],[8,9,10]]
# }

"""
String -> sequance of character
"Python" :->  P  y  t  h  o  n
              0  1  2  3  4  5 -> Positive indixing
             -6 -5 -4 -3 -2 -1
"""
# print(dic["ITI"]) # Learn
# print(dic["ITI"][-1]) # n
# print(dic[1][1]) # 5
# print(dic[3][0][1]) # 4

# dic["ITI"] = "Learn"
# "Learn":->[-1]
# [-1] = 'n'

# dic = {
#     int : [1,2,3],
#     2: 40,
#     2: 50
# }
# """
# key    value
# 2   ->  50
# """
# print(dic[2])
# # get method
# print(dic.get(2))
# print(dic.get(int))

# dic = {'x':11 , 'b':22 , 'y': 30}
# dic['a'] = 33
# print(dic)
# #
# # print(dic.pop('y'))

# while dic:
#     print(dic.popitem())
# print(dic)

# # clear -> dic.clear() -> remove all keys
# print(dic)
# print(dic.clear())
# print(dic)

# set -> () -> not repeated any values and sorted this values
# lst = [1,2,1,4,2,6,4,5,1,8,2,8,9,8,4,9]
# st = set(lst)
# print(st)

# lst = [1,2,1,4,2,6,4,5,1,8,2,8,9,"ITI",8,4,9,"ITI"]
# st = set(lst)
# print(st)

# x = '1'
# print(type(x)) #:-> String
# int_x = int(x)
# print(int_x, type(int_x))

# inp1,inp2 = map(int,input("Enter number: ").split())
# print(inp1,inp2)
# print(type(inp1),type(inp2))










# def to_upper_case(s):
#     return str(s).upper()

# def print_iterator(it):
#     for x in it:
#         print(x, end=' ')
#     print('')

# if __name__ == "__main__":
#     words = ["hello", "world", "python"]

#     upper_words = map(to_upper_case, words)

#     print_iterator(upper_words)

# def op(itm):
#     return itm*2
# if __name__ == "__main__":
#     lst= [1,2,3,4]
#     oper = map(op, lst)
#     print(list(oper))