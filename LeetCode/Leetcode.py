
# coding: utf-8

# # Q1)

# In[1]:


string = "parthoza"


# In[2]:


def isunique(s):
    
    if len(s) > 128: return False
    
    arr = [False]*128
    
    for char in s:
        
        if arr[ord(char)] is False:
            arr[ord(char)]= True
        else: return False
        
    return True


# # Q3)

# In[230]:


string = "a b c "


# In[21]:


def replacespace(s):
    
    st = []
    
    for char in s:
        
        st.append((char, '%20')[char == " "])
        
    return "".join(st)


# In[22]:


replacespace(string)


# # Q4)

# In[78]:


def checkpalidrom(string):
    counter = {}

    for char in string:

        try:
            counter[char] = counter[char] + 1

        except:

            counter[char] = 1

    return sum(v % 2 for v in counter.values()) <= 1


# # Q5)

# In[136]:


def strcompress(s):
    
    d = {}
    items = []
    
    for char in s:
        
        if char in d:
            d[char] = d[char] + 1
        
        else:
            d[char] = 1
    
    for k,v in d.items():
        items.append('{}{}'.format(k, v))
        
    return "".join(items)


# In[138]:


strcompress("dfs")


# # Fibonacci Series

# In[9]:


cache = {}

def fibo(n):
    
    if n in cache:
        return cache[n]
    
    if n == 1: value = 1
    
    elif n == 2: value = 2
        
    elif n > 2:
        
        value = fibo(n - 1) + fibo(n - 2)
    
    cache[n] = value
    
    return value


# In[10]:


get_ipython().magic('timeit fibo(1000)')


# # Prime Number

# In[34]:


import math
def isprime(n):
    
    if n == 1: return False
    elif n == 2: return True
    
    max_value = math.floor(math.sqrt(n))
    
    for d in range(3, max_value + 1, 2):
        
        if n % d == 0:
            return False
    
    return True


# # Delete duplicate from unsorted list

# In[332]:


L = [1, 2, 4, 2, 6]


# In[337]:


def del_dup(l):
    
    cache = {}

    for items in l:
        
        if items not in cache:
            
            cache[items] = "1"            

    return list(cache.keys())


# In[338]:


del_dup([10,9,1,2,3,4,6,5,5,5,5,2,1])


# # Given a list find pairs which are equal to given sum

# In[172]:


L = [7,5,7,5]
sum_of_nums = 12

def find_pairs(L, sum_of_nums):
    
    cache = {}
    count = 0
    for items in L:
        
        if sum_of_nums - items in cache:
            
            cache[sum_of_nums - items].add(items)
        
        elif items not in cache:
            
            cache[sum_of_nums - items] = {items}
        
        
        else:
            cache[items].add(items)
    
    return cache


# In[173]:


find_pairs(L,sum_of_nums)


# In[269]:


b = "tmmtww"
d = {}
count = 0
l = []
for num, char in enumerate(b, 1):
    
    if char not in d:
        
        d[char] = 1
        count = count + 1
        l.append(count)
        
    elif char in d and d[char] + 1 == num:
        
        l.append(count)
        count = 1
    
    elif char in d and d[char] + 1 != num and num != len(b):
        
        count = count +1
        l.append(count)
        count =  1
        
    elif char in d and num == len(b):
        l.append(count)
max(l)


# In[272]:


[sublist for items in [[1,2],[3,4]] for sublist in items]


# # FizzBuzz

# In[13]:


def fizzBuzz(n):
    # Write your code here
    if i % 2 == 0:
        return i

    elif i % 3 == 0:
        if i % 5 == 0:
            return 'FizzBuzz'
        return 'Fizz'

    elif i % 5 == 0:
        return 'Buzz'

    else: return i



# In[47]:


def find_str(s,t):
    
    s_1 = s.lower().strip(" ").split()
    t_1 = t.lower().strip(" ").split()

    union = set(s_1) & set(t_1)
    
    s = s.strip(" ").split()
    
    return [x for x in s if x.lower() not in list(union)]


# In[53]:


find_str("I am using HackerRank to improve programming","am HackerRank to improve")


# In[79]:


def find_str(s,t):
    l = re.sub(t, s, re.IGNORECASE).group()
    s_1 = s.strip(" ").split()
    s_1.remove(l)
    return s_1


# In[50]:


s="III"
cache = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

current_sum = 0
previous_sum = 0

for i in s[::-1]:
    current_sum = (current_sum + cache[i],current_sum - cache[i])[previous_sum > cache[i]]
    previous_sum = cache[i]
current_sum


# # Find longest same seq

# In[135]:


strs = ["flower","flight","flow"]
#strs = ['aa','aab']


# In[138]:


s_word = min(strs, key = len)

for i in range(len(s_word)):
    if len(set([x[i] for x in strs])) == 1:
        cache = {}
        cache[strs[0][:i+1]] = 1

"".join(cache.keys())


# # Find triplets with sum = 0

# In[225]:


L = [-1, 0, 1, 2, -1, -4]
for i in L:
    


# # Paranthesis

# In[381]:


s = "())("
L = []

if len(s) % 2 != 0 or s == "" or s == " ":
    pass

for i in s:
    if i == '(' or i=='{' or i=='[':
        L.append(i)
        
    elif L == []: print(False)
    
    elif i == ')' and L[-1] == '(':
        del(L[-1])
    
    elif i == ']' and L[-1] == '[':
        del(L[-1])
    
    elif i == '}' and L[-1] =='{':
        del(L[-1])


# # Duplicate in sorted array

# In[6]:


nums = [0,0,1,1,1,2,2,3,3,4]
j=0
for num in nums[1:]:
    
    if num != nums[j]:
        j += 1
        nums[j] = num


# # Needle and Haystack

# In[516]:


h = "mississippi"
n="issip"
if n in h:
    e = h.replace(n,"1")
e


# # The Social Network

# In[1]:


L = [2,1,1,2,1]
cache = {}

for x,i in enumerate(L):
    
    if i not in cache:
        cache[i] = [str(x)]
        
    else:
        
        cache[i].append(str(x))

for k,v in cache.items():
    l = [v[i:i + k] for i in range(0, len(v), k)]

    for i in l:
        print(" ".join(i))


# # Longest Sub - Seq

# In[231]:


# s = "pwwkew"
# d = s
# s = list(s)
# cache = {1:[]}
# max_sum = 0
# key = 1
# for x, y in enumerate(d):
#     if y not in cache[key]:
#         cache[key].append(y)
    
#     else:
#         #index = max(loc for loc, val in enumerate(s) if val == y and loc < x)
#         index = d[:x].rindex(y)
#         #print(index, index1)
#         key += 1
        
# #         if x - index == 0:
# #             cache[key] = [item]
    
        
# #         else:
#         cache[key] = s[index+1:x + 1]
#     max_sum = max(max_sum, len(cache[key]))

# cache



# LeetCode DP solution
s = "pwwkew"
max_len = 0
a = []

for letter in s: 
    if letter in a:
        a = a[a.index(letter)+1:]
    
    a.append(letter)
    print(a)
    max_len = max(max_len, len(a))
        
max_len


# # String to Int

# In[6]:


s = " --43cdc"
s = s.strip()


# # Plus One

# In[26]:


s = [1,2,3,4]
s = int(''.join(map(str,s))) + 1
[int(s) for s in str(s)]


# # First Unique Character in String

# In[ ]:


from collections import OrderedDict
class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        cache = OrderedDict([])
        
        if not s: return -1
                
        for x, i in enumerate(s):
            
            if i in cache:
                cache[i] = "False"
            else:
                cache[i] = x
        
        for k,v in cache.items():
            if v != "False":
                return v
        
        return -1


# # Climb Stairs

# In[77]:


class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        cache = {}

        if n in cache:
            return cache[n]

        if n == 1: value = 1
        elif n == 2: value = 2

        else:
            user = Solution()
            user1 = Solution()
            value = user.climbStairs(n - 1) + user1.climbStairs(n - 2)
            cache[n] = value

        return value
Solution().climbStairs(35)


# In[142]:


cache = {}

def climbStairs(n):
    """
    :type n: int
    :rtype: int
    """


    if n in cache:
        return cache[n]

    if n == 1: value = 1
    elif n == 2: value = 2

    else:
        value = climbStairs(n - 1) + climbStairs(n - 2)
    cache[n] = value

    return value
climbStairs(4)


# # Valid Palidrome

# In[95]:


import re
s = "A man, a plan, a canal: Panama"
s = s.lower().strip()
s = re.sub(r'\W','',s)
s == s[::-1]


# # Single Number

# In[269]:


nums = [2,2,3]
xor = 0
for i in nums:
    xor ^= i
xor


# # Vaid Palindrome

# In[1]:


s = "nl"
t = "cx"
product_1 = 1
product_2 = 1

for x,y in zip(s,t):
    product_1 *= ord(x)
    product_2 *= ord(y)


# # Get Ticket

# In[586]:


a = open('ticket.txt','r')
w = a.readlines()


# In[587]:


w = [x.strip('\n') for x in w]


# In[588]:


no_of_events = int(w[1])
no_of_buyers = int(w[2 + no_of_events])


# In[589]:


def manhattan(L):
    x1,y1,x2,y2 = map(int,L)
    return abs(x1 - x2) + abs(y1 - y2)


# In[607]:


events = {}
buyers = {}
for i in range(no_of_events, 2*no_of_events):
    events[w[i].split()[0]] = w[i].split()[1:]
    
for x, y in enumerate(w[2 + no_of_events + 1:], 1):
    buyers[x] = (y.split())
    
buyers_cop = buyers.copy()
events_cop = events.copy()


# In[591]:


events.keys()


# In[595]:


for i in buyers_cop.values():
    
    nearest = {}
    
    events_cop = {k:v for k,v in events_cop.items() if len(v) > 2}
    
    if len(events_cop) == 0: print('0,1'); break
        
    for x, y in events_cop.items():
        s = manhattan(i+y[:2])
        nearest[int(x)] = s
        
    print(nearest)  
    near_event = min(nearest, key = nearest.get)
    
    if len(events_cop[str(near_event)]) > 2:
        ticket = min(events_cop[str(near_event)][2:])
        print(near_event, ticket)
        events_cop[str(near_event)].remove(str(ticket))
        
    else:
        print('{} {}'.format('0','1'))


# # Website Pagination

# In[73]:


l = [[1,20,"Parth"], [10,20,"Dhairya"],[100,20,"Dhairya"],[11,20,"Dhairya"]]
sorted(l, key = lambda x: x[0], reverse = False)


# # Smart Sale

# In[68]:


L = [1,1,2,2]

m = 2

cache = {k:v for k,v in zip(list(set(L)), [L.count(x) for x in set(L)])}

for i in range(m):
    
    near = sorted(cache, key = cache.get)[0]
    
    cache[near] -= 1
    
    if cache[near] == 0:
        del(cache[near])
        
print(len(cache.keys()))


# # Happy Numbers

# In[12]:


n = 999
l = [int(x) for x in list(str(n))]
value = sum([x**2 for x in l])

while value > 0:
    
    value = sum([x**2 for x in l])
    l = [int(x) for x in list(str(value))]
    print(value, l)


# # Sieve of Atkin

# In[128]:


import math

def sieveOfAtkin(limit):
    P = [2,3]
    sieve=[False]*(limit+1)
    sieve[1], sieve[2] = True, True
    for x in range(1,int(math.sqrt(limit))+1):
        for y in range(1,int(math.sqrt(limit))+1):
            n = 4*x**2 + y**2
            if n<=limit and (n%12==1 or n%12==5) : sieve[n] = not sieve[n]
            n = 3*x**2+y**2
            if n<= limit and n%12==7 : sieve[n] = not sieve[n]
            n = 3*x**2 - y**2
            if x>y and n<=limit and n%12==11 : sieve[n] = not sieve[n]
    for x in range(5,int(math.sqrt(limit))):
        if sieve[x]:
            for y in range(x**2,limit+1,x**2):
                sieve[y] = False
    for p in range(5,limit):
        if sieve[p] : P.append(p)
    return len(P)


# In[129]:


sieveOfAtkin(10000)


# # Sieve of Eratosthenes - Finding Primes Python

# In[242]:


def eratosthenes(n):
    multiples = set()
    count = 0
    for i in range(2, n):
        if i not in multiples:
            count += 1
            for j in range(i*i, n+1, i):
                multiples.add(j)
    return count


# In[248]:


get_ipython().magic('timeit (eratosthenes(1500000))')


# In[407]:


def simpleSieve(sieveSize):
    #creating Sieve.
    sieve = [True] * (sieveSize+1)
    # 0 and 1 are not considered prime.
    sieve[0] = False
    sieve[1] = False
    for i in range(2,int(math.sqrt(sieveSize))+1):
        if sieve[i] == False:
            continue
        for pointer in range(i**2, sieveSize+1, i):
            sieve[pointer] = False
    # Sieve is left with prime numbers == True
    primes = []
    for i in range(sieveSize+1):
        if sieve[i] == True:
            primes.append(i)
    return len(primes)

get_ipython().magic('timeit (simpleSieve(1500000))')


# # Is Power of 3?

# In[466]:


def isPowerOfThree(self, n):
    """
    :type n: int
    :rtype: bool
    """
    if n == 0 or n < 0:return False
    return math.log10(n)/math.log10(3) % 1 == 0


# # Find missing numbers

# In[483]:


nums = [4,3,2,7,8,2,3,1]
for i in range(len(nums)):
    x = abs(nums[i])
    nums[x-1] = -1*abs(nums[x-1])
[i+1 for i in range(len(nums)) if nums[i]>0]


# # Cipher Text

# In[115]:


prev_int = 0
prev_str = ''
s = list('x1 = y2')
for x, y in enumerate(s):
    
    if re.findall('[aA-zZ]', y) and y.lower() in mapping and prev_str == '':
        prev_str = y
        continue
        
    if re.findall('\d', y) and prev_int == 0:
        prev_int = 9 - int(y)
        continue
        
    #After initializing
    
    if re.findall('[A-Z]', y):
        
        get_nbr = mapping[prev_str.lower()] + mapping[y.lower()]
        if get_nbr > 25:
            
            get_nbr = get_nbr - 26
            
            find_str = maps[get_nbr]
            s[x]= find_str.upper()
            prev_str = s[x]
        else:
            
            find_str = maps[get_nbr]
            s[x]= find_str.upper()
            prev_str = s[x]
            
            
    elif re.findall('[a-z]', y):
        get_nbr = mapping[prev_str.lower()] + mapping[y.lower()]
        
        if get_nbr > 25:
            
            get_nbr = get_nbr - 26
            
            find_str = maps[get_nbr]
            s[x]= find_str
            prev_str = s[x],
            
        else:
            find_str = maps[get_nbr]
            s[x]= find_str
            prev_str = s[x]
    
    elif re.findall('\d', y):
        get_nbr = (int(y) + prev_int) % 10
        s[x] = str(get_nbr)
        prev_int = get_nbr
''.join(s)


# # Mine Sweeper

# In[122]:


game = [['.','.','.','m'],
       ['.','.','.','.'],
        ['.','m','.','.']]


# # Strengthen Passwords

# In[301]:


s = ['SSSS']

for x, y in enumerate(s):
    
#     if re.findall(r's',y,re.IGNORECASE):
#         y = re.sub(r's|S','5',y,re.IGNORECASE)
#         s[x] = y
    y = y.replace('s','5')
    y = y.replace('S','5')
    s[x] = y
    
    if len(y) % 2 != 0:
        
        if len(y) > 1 and re.findall('\d',y[len(y)//2]):
            t = list(y)
            nbr = re.findall('\d',t[len(t)//2])[0]
            #y = y.replace(nbr, str(int(nbr)*2))
            t[len(t)//2] = str(int(nbr)*2)
            s[x] = ''.join(t)
    
    if len(y) % 2 == 0:
        t = list(y)
        t[0], t[-1] = t[-1].swapcase(), t[0].swapcase()
        s[x] = ''.join(t)
            
    if re.findall(r'nextcapital', y, re.IGNORECASE):
        z = re.search(r'next', y, re.IGNORECASE).group()
        y = y.replace(z,z[::-1])
        s[x] = y
        

s


# In[283]:


s = "SS5"
s[len(s)//2]


# # Remove Substring

# In[41]:


new = []
i = 0
k = 0
while i < len(s):
    
    if k < len(t) and s[i] == t[k] :
        i += 1
        k += 1
    else:
        new.append(s[i])
        i += 1

new


# # Find plane seats

# In[213]:


rows = 2
seats = '1A 2F 1C'
seats = list(seats.split(' '))
family = rows*3

seats = sorted(seats, key = lambda x: x[0])

L = []

for i in range(1,rows + 1):
    L.append([str(i)+chr(x) for x in range(65,76) if x != 73])

L = [j for i in L for j in i]

for i in seats:
    
    if i in L:
        L[L.index(i)] = '0'
        
x = 0
while x <len(L):
    
    if '0' in (L[x], L[x+1],L[x+2]):
        family -= 1
    
    if '0' in (L[x+3],L[x+4],L[x+5]) or '0' in (L[x+4],L[x+5],L[x+6]):
        family -= 1
            
#     if '0' in (L[x+4],L[x+5],L[x+6]):
#         if '0' in (L[x+3],L[x+4],L[x+5]):
#             family -= 1
    
    
    if '0' in (L[x+7], L[x+8],L[x+9]):
        family -= 1
        
    x=x+10
    
family


# # Trailing zeros in Factorial

# In[24]:


def trailingZeroes(n):
    """
    :type n: int
    :rtype: int
    """

    r = 0
    while n > 0:
        n //= 5
        r += n
    return r


# In[25]:


m = [[1,2,3,5], [1,7,8,9], [3,4,5,1], [19,20,1,2]]

m = [set(x) for x in m]

s = m[0]

for i in m[1:]:
    k = s.intersection(s, i)
    s = k
s


# In[28]:


m = [[1,2,3,5], [1,7,8,9], [3,4,5,1], [19,20,1,2]]
cache = {x:1 for x in m[0]}

for i in m[1:]:
    for items in i:
        if items in cache:
            cache[items] += 1
cache


# # Most frequent substring

# In[11]:


def strings(s,min_length,max_length,unique):
    
    if len(s) == len(set(s)): return 1
    
    combinations = [s[i:j+1] for i in range(len(s)) for j in range(i,len(s)) if max_length >= len(s[i:j+1]) >= min_length]
    
    unique_combinations = set(combinations)
    
    max_len = 0
    for i in unique_combinations:
        temp = combinations.count(i)
        if temp > max_len:
            max_len = temp
    return max_length


# # Email Id Groups (Google)

# In[98]:


L = ['a.b@example.com','ab+1@example.com','x@example.com','x@exa.mple.com','y@example.com','y@example.com','y@example.com']
cache = {}
for i in L:
    
    p = i.split('@')
    p1 = p[0]
    p2 = p[1]
    
    if '.' in p1:
        p1 = re.sub(r'\.','',p1)
        
    elif '+' in p1:
        p1 = p1.split('+')[0]
    
    x = ''.join(p1 + p2)
    
    if x in cache:
        cache[x] += 1
        
    else:
        cache[x] = 1
        
print(len([x for x in d.values() if x >= 2]))


# # Maximum sub - contiguous array

# In[31]:


l = [-2,1,-3,4,-1,2,1,-5,4]

curr = maxi = l[0]

for items in l[1:]:
    
    curr = max(items, curr + items)
    maxi = max(maxi, curr)
    print(curr, maxi)

maxi


# # Rotate Array

# In[53]:


l = [1,2,3,4,5,6,7]
k = 3
#l[-k:], l[:-k] = l[:-k], l[-k:]   
l[:-k], l[-k:] = l[-k:], l[:-k]
l


# # Roberry

# In[52]:


l = [7, 2, 9, 3, 1]
dp = [0] * len(l)

dp[0] = l[0]
dp[1] = max(l[0], l[1])

for i in range(2, len(l)):
    
    dp[i] = max(l[i] + dp[i - 2], dp[i - 1])

dp


# # Move Zeros

# In[133]:


l = [0,1,0,3,12]
l.sort(key = lambda b: -1 if b == 0 else 0, reverse = True)
l


# # Excel Sheet Column Number

# In[183]:


cache = {chr(i):x for i, x in zip(range(65, 91), range(1, 27))}

s = 'ZZ'

if len(s) == 1:print(cache[s[0]])

pro = 0
for x, y in enumerate(s, 1):
    pro += 26 ** (len(s) - x) * cache[y]

pro


# # Excel Sheet Column Title

# In[186]:


701 % 26


# # Buy and Sell stock

# In[79]:


l = [1,2,6,4]
buy = min(l)
profit = []
while buy != max(l):
    
    for i in l[l.index(buy) + 1:]:
        if i > buy:
            profit.append(i - buy)
    
    l.remove(buy)
    buy = min(l)

if len(profit) == 0:
    print(0)


# In[80]:


prices = [7,1,5,3,6,4]
profit = []
for i in prices:
    temp = i
    p = [x - temp for x in prices[prices.index(i) + 1:] if x > temp]
    if len(p) != 0:
        profit.append(max(p))
print(max(profit) if len(profit) != 0 else 0)


# In[44]:


prices = [7,1,5,3,6,4]

max_profit ,  min_price = 0, float('inf')

for i in prices:
    
    min_price = min(min_price, i)
    profit = i - min_price
    max_profit = max(profit, max_profit)
    
max_profit


# # Pascals Triangle

# In[140]:


n = 15
L = [[1]]
for i in range(2, n + 1):
    
    pascal = [0] * i
    pascal[0] = pascal[-1] = 1
    
    L.append(pascal)
    
for x, y in enumerate(L[2:], 1):
    for i in range(1, len(y) - 1):
        y[i] = L[x][i - 1] + L[x][i]
L


# # Find all anagrams

# In[231]:


from collections import Counter
s = 'eklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjdwtcaxzsnifvhmoueklpyqrbgjd'
p = 'yqrbgjdwtcaxzsnifvhmou'
hash_p = Counter(p)
L = []
for x, y in enumerate(s):
    
    if Counter(s[x:x+len(p)]) == hash_p:
        L.append(x)


# # Sliding Window Solution

# In[75]:


s = 'abcb'
p = 'abc'
res = []

pCounter = Counter(p)
sCounter = Counter(s[:len(p)-1])

for i in range(len(p)-1,len(s)):
    sCounter[s[i]] += 1
    if sCounter == pCounter:
        res.append(i - len(p) + 1)
    
    sCounter[s[i-len(p)+1]] -= 1
    
if sCounter[s[i-len(p)+1]] == 0:
    del sCounter[s[i-len(p)+1]]

res


# # Sort given 2 arrays O(n)

# In[1]:


arr1 = [1,2,3,3]
arr2 = [4,5,6]
arr3 = []
i = j = 0
while len(arr3) != len(arr1) + len(arr2):
    
    if i <= len(arr1) - 1 and j <= len(arr2) - 1 and arr1[i] <= arr2[j]:
        arr3.append(arr1[i])
        i += 1
    
    else:
        arr3.append(arr2[j])
        j += 1
arr3


# # Roblox

# In[340]:


cache = {'Jan':'01', 'Feb':'02','Mar':'03','Apr':'04','Jun':'05', 'Jul':'06', 'Aug':'08', 'Sep':'09','Oct':'10','Nov':'10','Dec':'12'}
s = '20th Oct 2052'
s = s.strip().split()
ans = []
date = re.findall('\d+',s[0])[0]
print(date)
s[0] = date.zfill(2) if len(date) < 2 else date
s[1] = cache[s[1]]
s = s[::-1]
ans.append('-'.join(s))


# In[342]:


def get_all_substrings(input_string, minLength, maxLength, maxUnique):
    
    cache = {}
    length = len(input_string)
    combinations =  [input_string[i:j+1] for i in range(length) for j in range(i,maxLength) if j - i <= maxLength and 
                     len(input_string[i:j+1]) >= minLength
           and len(input_string[i:j+1]) <= maxLength and len(set(input_string[i:j+1])) <= maxUnique]
    
    for items in combinations:
        if items not in cache:
            cache[items] = 1
        else:
            cache[items] += 1
    return cache
get_all_substrings('ababab',2,4,3)


# # Pathrise

# In[419]:


a = 'ab'
b = 'zsd'

if len(a) == len(b):
    s = [a[i] + b[i] for i in range(len(a))]
    
else:
    s = []
    x = 0
    cache = max(a,b, key = len)   
    cache_min = min(len(a), len(b))
    cache_max = max(len(a), len(b))
    
    while x != cache_max:
        if x < cache_min:
            s.append(a[x])
            s.append(b[x])
            x += 1
            
        else:
            s.append(cache[x])
            x += 1
''.join(s)


# # Shortest unsorted continuous array

# In[60]:


#L = [2, 6, 4, 8, 10, 9, 15]
L = [1, 3, 2, 3, 3]
L = [3,2]
cache = []
x = 0
min_index, max_index = float('inf'),float('inf')
while sorted(L) != L:
    for x in range(len(L) - 1):

        if L[x + 1] <  L[x]:
            cache.append(x)
            cache.append(x+1)
            L[x], L[x + 1] = L[x + 1], L[x]
            x += 1
len(L[min(cache):max(cache) + 1])


# In[106]:


L = [1,3,2,2,2]
cache = sorted(nums)
r, l = min(i for i in range(len(L)) if nums[i] != cache[i]), max(i for i in range(len(L)) if nums[i] != cache[i])
l - r + 1


# # Hamming Distance

# In[140]:


x = 1
y = 4
get_bin = lambda x: '{0:b}'.format(x)
x, y = get_bin(x), get_bin(y)
padding = int(max(len(x), len(y)))
x = x.zfill(padding)
y = y.zfill(padding)
count = 0
for a, b in zip(x, y):
    if a!=b:
        count+=1
count


# # Two Sum - II

# In[167]:


numbers = [2, 7, 11, 15]
target = 9
indexes = []
for items in range(len(numbers)):
    
    if target - numbers[items] in numbers and len(indexes) < 2:
        indexes.append(items)
        get = [x for x, y in enumerate(numbers) if x > indexes[0] and y == target - numbers[items]]
        indexes.append(get[0])
indexes


# In[194]:



numbers = [2, 7, 11, 15]
target = 9
cache = {}
ans = []
for x, items in enumerate(numbers):

    if target - items in cache:

        cache[target - items].append((items, x))

    elif items not in cache:

        cache[target - items] = [(items, x)]


    else:
        cache[items].append((items, x))

for k, v in cache.items():
    if len(v) == 2 and v[0][0] + v[1][0] == target:
        ans.append(v[0][1] + 1) # 1 - Index based
        ans.append(v[1][1] + 1) # 1- Index based
ans


# # Intersection of 2 arrays

# In[208]:


nums1 = [4,4,5,9,9]
nums2 = [9,4,9,8,4]
ans = []
cache = list(set(nums1) & set(nums2))
for items in cache:
    
    ans.append([items] * min(nums1.count(items), nums2.count(items)))

[i for i in ans for i in i]


# # Final Discounted Price (Courseera)

# In[136]:


L = [1,3,3,2,5]
ans = L[-1]
curr = 0
for items in range(len(L) - 1):
    curr = L[items]


# In[142]:


a = [1,1,0,1,1,1]
a = ''.join(map(str,a))
len(max(a.split('0'), key = len))


# # Top k frequent elements

# In[22]:


import collections
import heapq
# counts = collections.Counter(nums)
k = 2
nums = [1,1,1,1,5,3,2,6,7,6]

d = {}
for i in nums:
    if i not in d:
        d[i] = 1
    else:
        d[i] += 1

heap = []
for num, count in d.items():
    heapq.heappush(heap,(count, num))
    if len(heap) > k :
        heapq.heappop(heap)

res = []
for _ in range(k):
    res.append(heapq.heappop(heap)[1])
res


# In[56]:


from collections import Counter
L = [1,1,1,2,2,3]
[x[0] for x in Counter(L).most_common(2)]


# # Subsets

# In[29]:


nums = [1,2,3]
res = [[]]
for num in sorted(nums):
    res += [item + [num] for item in res]
res


# # Subsets 2

# In[32]:


nums = [1,2,2]
res = [[]]
for num in sorted(nums):
    res += [item + [num] for item in res if (item + [num]) not in res]
res


# # Wiggle Sort

# In[135]:


nums = [1,5,5,4,3]
arr = sorted(nums)
for i in range(1, len(nums), 2): nums[i] = arr.pop() 
for i in range(0, len(nums), 2): nums[i] = arr.pop() 
nums


# # IMC

# In[156]:


#Create ships
s = '1A 1B'
s = s.split(',')
s = [x.split(' ') for x in s]
for x, items in enumerate(s):
    nums = abs(int(re.findall('\d+',items[0])[0]) - int(re.findall('\d+',items[-1])[0])) + 1
    chars = abs(ord(re.findall('[A-Z]',items[0])[0]) - ord(re.findall('[A-Z]',items[-1])[0])) + 1
    s[x] = [nums * chars] + items
cache = copy.deepcopy(s)

#Create hits
t = '1C'
t = t.split(' ')
hit_sunk, hit_not_sunk = 0, 0

for nums, items in enumerate(s):
    for hits in t:
        if check_hit(items, hits) is True:
            cache[nums][0] -= 1


for x, y in zip(cache, s):
    
    if x[0] == 0:
        hit_sunk += 1
    
    elif x[0] < y[0]:
        hit_not_sunk += 1

print(hit_sunk, hit_not_sunk)


# In[144]:


def check_hit(items, hit):
    x, y = re.findall('[A-Z]',items[1])[0], re.findall('[A-Z]',items[-1])[0]
    xx, yy = int(re.findall('\d+',items[1])[0]), int(re.findall('\d+',items[-1])[0])
    
    hit_x = re.findall('[A-Z]',hit)[0]
    hit_xx = int(re.findall('\d+', hit)[0])
    
    if xx <= hit_xx <= yy and x <= hit_x <= y:
        return True
    
    else:
        return False


# In[228]:


A = [1,1]
n = len(A)
i = n - 1
result = -1
maximal = 0
k = 0
while (i > 0):
    if (A[i] == 1):
        k = k + 1
        if (k >= maximal):
            maximal = k
            result = i
    else:
        k = 0
    i = i - 1
if (A[i] == 1 and k + 1 >= maximal):
    result = 0
result


# # Unique Path

# In[305]:


import math
m = 7 #columns
n = 3 #rows
# Can move only right or down
math.factorial((m - 1) + (n - 1))/(math.factorial(m - 1) * math.factorial((m - 1) + (n - 1) - (m - 1)))


# # Permutations

# In[389]:


nums = [1,2,3]

def permute(nums):
    
    def backtrack(start, end):
        
        if start == end:
            ans.append(nums[:])
            
        for i in range(start, end):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start+1, end)
            nums[start], nums[i] = nums[i], nums[start]

    ans = []
    backtrack(0, len(nums))
    
    return ans


# In[94]:


L = [1,2,3]
def backtrack(nums, start, end):

    if start == end:
        print(nums)
        
    for i in range(start, end + 1):
        nums[start], nums[i] = nums[i], nums[start]
        backtrack(nums, start + 1, end)
        nums[start], nums[i] = nums[i], nums[start]
        
backtrack([1,2,1], 0 , 2)


# In[248]:


nums = [1,2,3]
ans = [[]]
for x in nums:
    ans = [items + [n] for items in ans for n in nums if (n not in items)]
ans


# In[315]:


nums = [1, 2, 3]
ans = [[]]

for x in nums:
    result = []
    for items in ans:
        for n in nums:
            if n not in items:
                result.append(items + [n])
    ans = result
print(ans)


# # Permutations 2

# In[391]:


from collections import Counter
nums = [1,1,2]
ans = [[]]

cache = Counter(nums)

for idx, x in enumerate(nums):
    result = []
    for items in ans:
        cache1 = Counter(items)
        for id, n in enumerate(nums):
            if cache[n] != cache1[n] and items + [n] not in result:
                result.append(items + [n])

    ans = result
ans


# # Sort colors

# In[100]:


#Method 1
nums = [2,0,2,1,1,0]
zero,one,two = nums.count(0) - 1, nums.count(1) - 1, nums.count(2) - 1

for items in range(len(nums)):
    
    if zero >= 0:
        nums[items] = 0
        zero -= 1
        
    elif one >= 0:
        nums[items] = 1
        one -= 1
        
    elif two >= 0:
        nums[items] = 2
        two -= 1
nums


# In[101]:


# Method 2
zero, one, two = 0, 0, len(nums) - 1
while one <= two:
    if nums[one] == 0:
        nums[zero], nums[one] = nums[one], nums[zero]
        zero += 1
        one += 1
    
    elif nums[one] == 1:
        one += 1
    
    else:
        nums[one], nums[two] = nums[two], nums[one]
        two -= 1
nums


# # 3 sum

# In[46]:


nums = [-1, 0, 1, 2, -1, -4]
ans = []
nums = sorted(nums)

for items in nums:

    target = -1 * items 
    temp = nums[nums.index(items) + 1:]
    start, end = 0, len(temp) - 1
    while start < end:

        if temp[start] + temp[end] == target:
            ans.append((temp[start], temp[end], items))
            start += 1
            end -= 1
            
        elif temp[start] + temp[end] > target:
            end -= 1

        else:
            start += 1
ans


# In[49]:


list(map(list, set(ans)))


# # Group Anagrams

# In[171]:


strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
cache = {}

for items in strs:
    if ''.join(sorted(items)) not in cache:
        cache[''.join(sorted(items))] = [items]
    else:
        cache[''.join(sorted(items))].append(items)
[x for x in cache.values()]


# # Reconstruct Iternary

# In[359]:


L = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
iternary = [x for x in L if x[0] == 'JFK']
del(L[L.index(iternary[0])])
curr = 0
while L:
    temp = [x for x in L if x[0] == iternary[curr][1]]
    iternary += temp
    del(L[L.index(temp[0])])
    curr += 1
iternary


# In[404]:


L = [["JFK","KUL"],["JFK","NRT"],["NRT","JFK"]]
#L = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
iternary = [x for x in L if x[0] == 'JFK']
if len(iternary) > 1:
    iternary.sort(key = lambda x:x[1])
    del(iternary[-1])
    del(L[L.index(iternary[0])])
else:
    del(L[L.index(iternary[0])])


curr = 0

while L:
    temp = [x for x in L if x[0] == iternary[curr][1]]
    
    if len(temp) > 1:
        temp.sort(key = lambda x:x[1])
        del(temp[-1])
    
    iternary += temp
    curr += 1
    del(L[L.index(temp[0])])


[x[0] for x in iternary] + [iternary[-1][-1]]


# # Course Schedule 3

# In[6]:


import heapq
pq = []
start = 0
A = [[100, 200], [200, 1300], [1000, 1250], [2000, 3200]]
for t, end in sorted(A, key = lambda x: x[1]):
    start += t
    heapq.heappush(pq, -t)
while start > end: 
    start += heapq.heappop(pq)
pq


# In[ ]:


pq = []
 start = 0
 for t, end in sorted(A, key = lambda (t, end): end):
     start += t
     heapq.heappush(pq, -t)
     while start > end:
         start += heapq.heappop(pq)
 return len(pq)


# # Longest palindromic substring

# In[53]:


s = "mkuixwymlzmvrtxpqtomvvpsdnwgslqhyqopwzlgkktjxpayeratkvetdzbevkfkckmmjvftrrbmykvvklzjve"
#if len(set(s)) == 1 or not s: return s
#s = "babad"

def check_palindrome(s):
    
    start, end = 0, -1

    if len(s) < 3:
        return True if s[0] == s[-1] else False

    elif s[start] == s[end]:
        cache = check_palindrome(s[start + 1:end])
        return cache
    
    else:
        return False
        break

    
ans = [s[i:j + 1] for i in range(len(s)) for j in range(i,len(s)) if s[i] == s[j]]
#ans = (x for x in ans if x == x[::-1])
answer = ''
for x in ans:
    
    if len(x) <= 3:
        if x[0] == x[-1]:
            answer = max(answer, x, key = len)
    
    else:
        if check_palindrome(x[1:-1]):
            answer = max(answer, x, key = len)
answer


# In[3]:


import numpy as np
X = [[3, 'aa', 10],                 
     [1, 'bb', 22],                      
     [2, 'cc', 28],                      
     [5, 'bb', 32],                      
     [4, 'cc', 32]]
# X is a list of list
X = np.array(X)

X


# # Container With Most Water

# In[509]:


num = [1,8,6,2,5,4,8,3,7]
area = 0
start = 0
#l, b = 0, 0
end = len(num) - 1
while start < end:
    if num[start] < num[end]:
        area = max(area, min(num[start], num[end]) * (end - start))
        #l, b = max(num[start], l), max(num[end], b)
        start += 1
        
    else:
        area = max(area, min(num[start], num[end]) * (end - start))
        #l, b = max(num[start], l), max(num[end], b)
        end -= 1
area


# # Letter combinations of a Phone Number

# In[74]:


def letterCombinations(digits):
    """
    :type digits: str
    :rtype: List[str]
    """
    if not digits:return []
    cache = {'1':['*','*','*']}
    mapping = [chr(x) for x in range(97, 123)]
    curr = 0
    for i in range(2, 10):
        if i not in (7, 9):
            temp = mapping[curr:curr + 3]
            curr = mapping.index(temp[-1]) + 1
            cache[str(i)] = temp

        elif i == 9:
            temp = mapping[curr:curr + 3]
            curr = mapping.index(temp[-1]) + 1
            cache[str(i)] = mapping[22:]

        elif i == 7:
            temp = mapping[curr:curr + 4]
            curr = mapping.index(temp[-1]) + 1
            cache[str(i)] = temp

    if len(digits) == 1: return cache[digits]
    ans = [[]]

    for items in digits[:2]:
        ans += [cache[items]]
    del(ans[0])

    ans = [ans[0][i] + ans[z][j] for i in range(len(ans[0])) for z in range(1,len(ans)) for j in range(len(ans[z]))]

    if len(digits) > 2:

        for items in digits[2:]:
            temp = cache[items]
            ans = [x + n for x in ans for n in temp]

    return list(set(ans))
letterCombinations('17')


# In[76]:


digits = '17'
letters = [[],['*','*','*'],['a','b','c'],['d','e','f'],['g','h','i'],['j','k','l'],['m','n','o'],['p','q','r','s'],['t','u','v'],['w','x','y','z']]
res = letters[int(digits[0])]

for i in digits[1:]:
    get = int(i)
    res = [i + j for i in res for j in letters[get]]
list(set(res))


# # Generate Paranthesis

# In[86]:


n = 3
dp = {1: set(['()']), 2: set(['(())', '()()'])}
for i in range(3, n+1):
    # pattern 1: outer parenthese + subproblem with length - 1
    dp[i] = set(['(' + x + ')' for x in dp[i-1]])
    for j in range(1, i):
        # pattern 2: dp[i] is formed by dp[j] + dp[i-j]
        dp[i] = dp[i].union([x + y for x in dp[j] for y in dp[i-j]])
dp


# # Shortest Distance to Character

# In[44]:


S = "loveleetcode"
C = 'e'
indexes = [i for i, x in enumerate(S) if x == C]
ans = []
for i, x in enumerate(S):
    ans.append(min([abs(i - j) for j in indexes]))
ans


# # Flipping Game

# In[73]:


#fronts = [1,2,4,4,7]
#backs = [1,3,4,1,3]
fronts = [2,2,5,1,2]
backs = [4,1,2,1,1]
ori_front = fronts[:]
ori_backs = backs[:]
temp = float('inf')

while True:
    
    for x in range(len(fronts)):

        if fronts[x] == backs[x]:
            continue
        else:
            fronts[x],backs[x] = backs[x], fronts[x]
            if backs[x] not in fronts:
                print(backs[x])
                temp = min(temp, backs[x])
        print(fronts, backs)
    if ori_front == fronts and ori_backs == backs:
        break
temp


# # Find kth largest element

# In[12]:


nums = [1,2,3,4,5]
k = 5
for i in range(k-1):
    nums.pop()
nums[-1]


# # Largest Number

# In[189]:


nums = [20 , 1]
nums = [''.join(sorted(str(x), key = lambda x:str(x), reverse = True)) for x in nums]
s = ''
cache = []
for items in range(1, len(max(nums, key = len)) + 1):
    s += ''.join(map(str, sorted([x for x in nums if len(str(x)) == items],reverse = True)))
    cache.append(''.join(map(str, sorted([x for x in nums if len(str(x)) == items],reverse = True))))
cache.sort(key = lambda x:x[0],reverse = True)


# In[79]:


from functools import cmp_to_key
nums = [1,2,3]
a = list(map(str, nums))              
a.sort(key = cmp_to_key(lambda x, y: int(x+y) - int(y+x)), reverse = True)                
a


# # First and last position of sorted array

# In[337]:


nums = [3,3,3]
target = 3
temp = False
start = len(nums)//2
ans = []
if nums[0] > target or nums[-1] < target:print([-1,-1])

while start >= 0 and start < len(nums):
    
    if nums[start] == target:
        ans.append(start)
        if start + 1 < len(nums) and nums[start + 1] == target:
            ans.append(start + 1)
            start += 1
        elif start - 1 >= 0 and nums[start - 1] == target:
            ans.append(start - 1)
            start -= 1
            
    if nums[start] > target:
        start -= 1
        temp == True
    else:
        if temp == True:
            break
        start += 1
        temp -= 1
        
    #if len(ans) == 2:break

ans


# In[413]:


nums = [3,3,3,3,3]
start = len(nums) // 2
start_pt, end_pt = float('inf'), 0
target = 3

while start <= len(nums) - 1:
    if nums[start] > target:
        break
    if nums[start] == target:
        start_pt = min(start_pt, start)
        end_pt = max(end_pt, start)
        start += 1
    else:
        start += 1

start = len(nums) // 2

while start >= 0:
    if nums[start] < target:
        break
    
    if nums[start] == target:
        start_pt = min(start_pt, start)
        end_pt = max(end_pt, start)
        start -= 1
    else:
        start -= 1

start_pt, end_pt


# # Number of Island

# In[78]:


grid = [["1","1","1","1","0"],
        ["1","1","0","1","0"],
        ["1","1","0","0","0"],
        ["0","0","0","0","0"]]

def numIslands(grid):
    """
    :type grid: List[List[str]]
    :rtype: int
    """
    def df(i, j):

        if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':

            grid[i][j] = '0'

            list(map(df,(i + 1, i - 1, i, i), (j, j, j + 1, j - 1)))
            return 1

        return 0

    return sum(df(i, j) for i in range(len(grid)) for j in range(len(grid[i])))
numIslands(grid)


# # Word Search

# In[62]:


def exist(board, word):
    
    if not board:
        return False
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == word[0]:
                if dfs(board,i, j, word, 0):
                    return True
                return  False
            
def dfs(board, i, j, word, index):
    
    if len(word) == index:
        return True
    
    if i < 0 or i >= len(board) or j < 0  or j >= len(board[0]) or word[index] != board[i][j]:
        return False
    
    tmp = board[i][j]
    board[i][j] = '#'

    res = dfs(board, i+1, j, word, index + 1) or dfs(board, i-1, j, word, index + 1)     or dfs(board, i, j+1, word, index + 1) or dfs(board, i, j-1, word, index + 1)
    board[i][j] = tmp
    
    return res

board =[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

word = 'ABCCSEED'

exist(board, word)


# In[17]:


def exist(board, word):
    if not board:
        return False
    for i in range(len(board)):
        for j in range(len(board[0])):
            if dfs(board, i, j, word):
                return True
    return False

# check whether can find word, start at (i,j) position    

def dfs(board, i, j, word):
        
    if len(word) == 0: # all the characters are checked
        return True
    
    if i<0 or i>=len(board) or j<0 or j>=len(board[0]) or word[0]!=board[i][j]:
        return False
    
    
    tmp = board[i][j]  # first character is found, check the remaining part
    board[i][j] = "#"  # avoid visit agian 
    # check whether can find "word" along one direction
    res = dfs(board, i+1, j, word[1:]) or dfs(board, i-1, j, word[1:])     or dfs(board, i, j+1, word[1:]) or dfs(board, i, j-1, word[1:])
    board[i][j] = tmp
    return res

board =[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

word = 'ABCCSEED'

exist(board, word)


# In[9]:


word = 'ASFD'
for i in range(5):
    print(word[1:])


# # Search in sorted array

# In[554]:


def search(self, nums, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if not nums: return -1

    start, end = 0, len(nums) - 1
    #mid = len(nums)//2

    while start <= end:

        mid = (start + end)//2

        if nums[mid] == target:
            return mid

        if nums[start] <= nums[mid]:

            if nums[start] <= target <= nums[mid]:

                end = mid - 1

            else:

                start = mid + 1


        else:
            if nums[mid] <= target <= nums[end]:
                start = mid + 1

            else:
                end = mid - 1
    return -1


# # Symmetric Tree

# In[555]:


class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        if root:
            return self.traverse(root.left, root.right)
        return True
            
    def traverse(self, p, q):
        
        if p and q:
            return p.val == q.val and self.traverse(p.left, q.right) and self.traverse(p.right, q.left)
        return p is q


# # Jewels and Stones

# In[577]:


import re
J = "ebdf" 
S = "a"
cache = list(set(J) & set(S))
sum([S.count(x) for x in cache])


# # Product of Array Except Self

# In[168]:


nums = [2,3,4,5,6]
out = [None for x in range(len(nums))]

tmp = 1

for idx in range(len(nums)):
    out[idx] = tmp
    tmp *= nums[idx]

tmp = 1

for idx in range(len(nums)-1, -1, -1):
    out[idx] *= tmp
    tmp *= nums[idx]
out


# In[88]:


nums = [2,3,4,5,6]
size = len(nums)
prods = [1]*size

Lp = Rp = 1

for i in range(size):
    
    prods[i] *= Lp
    
    prods[~i] *= Rp
    
    Lp *= nums[i]
    
    Rp *= nums[~i]


# # Convert BST to Greater tree

# In[ ]:


class Solution:
    
    def __init__(self):
        self.cache = 0
    
    def convertBST(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        
        if root:
            self.convertBST(root.right)
            self.cache += root.val
            root.val = self.cache
            self.convertBST(root.left)
        return root
# All types of traversal
#https://www.geeksforgeeks.org/tree-traversals-inorder-preorder-and-postorder/


# # Perfect Squares

# In[435]:


n = 22

cache = [x**2 for x in range(1, n) if x**2 < n]
count = 0
while n != 0:
    
    n = n - math.floor(math.sqrt(n))**2
    count += 1
    if n == 0:
        break
        
count


# In[ ]:


dp = [float('inf') for _ in range(n+1)]
dp[0] = 0
dp[1] = 1

for i in range(2, n+1):
    for j in range(int(math.sqrt(n))+1):
        dp[i] = min(dp[i], dp[i-j*j]+1)
        print(dp[i], dp[i-j*j]+1, i, j)


# # Clarivate Analytics

# In[257]:


import random
import string
import heapq
import tracemalloc


def hIndex(citations):
    """
    O (logn)
    """
    
    n = len(citations) 
    start, end = 0, n-1
    
    while start <= end:
        
        middle = (start + end) // 2
        
        if citations[middle] == n - middle: return citations[middle]
        
        elif citations[middle] < n - middle: start = middle + 1
        
        else: end = middle - 1
    
    return n-end-1


def main():
    
    #Create generator for data
    #No memory allocated
    articles = ([''.join(random.choices(string.ascii_uppercase, k=10)),                          ''.join(random.choices(string.ascii_uppercase, k=1)),                          ''.join(random.choices(string.ascii_uppercase, k=25)),                          random.randint(0,5)] for _ in range(100000))

    
    # Initial Memory - O(n)
    cache = {}

    for x in range(100000):
        items = next(articles)

        if items[1] not in cache:
            cache[items[1]] = [items[-1]]

        else:
            heapq.heappush(cache[items[1]], items[-1])


    
    # O(n log n)
    cache.update((k, hIndex(cache[k])) for k in cache)
    
    
    # Get Top K
    k = 2

    # Additional Memory - O(k)
    heap = []

    for num, count in cache.items():
        heapq.heappush(heap,(count,num))
        if len(heap) > k :
            heapq.heappop(heap)

    top_k = []

    for _ in range(k):
        top_k.append(heapq.heappop(heap)[1])

    return  top_k[::-1]

#Run Time - O(n log n)
#Memory - O(k)

if __name__ == "__main__":
    get_ipython().magic('memit main()')


# # Course Schedule

# In[8]:


#prerequisites = [[1,0],[2,0],[3,1],[3,2]]
prerequisites = [[1, 0], [0, 1]]
path = set()
visited = set()
ans = set()

def leetcode(prerequisites):
    
    cache = {}
    
    for k, v in prerequisites:
        if v not in cache:
            cache[v] = [k]
        else:
            cache[v].append(k)
    
    return cache

def dfs(node):

    if node in visited:
        return False

    visited.add(node)    
    path.add(node)
    for prerequisite in cache.get(node, ()):
        if prerequisite in path or dfs(prerequisite):
            return True
    ans = visited
    path.remove(node)
    return False

cache = leetcode(prerequisites)
not any(list(dfs(i) for i in cache))


# # Maximum Product Sub Array

# In[251]:


nums = [-2,1,-2]
curr_max, curr_min = 1, 1
best = max(nums)

for n in nums:

    choices = [curr_max*n, curr_min*n, n]
    curr_max, curr_min = max(choices), min(choices)
    best = max(best, curr_max)  

best


# # Course Schedule 2

# In[11]:


numCourses =  4
prerequisites = [[1,0],[2,0],[3,1],[3,2]]

graph = {}

cache = set([x for x in prerequisites for x in x])
for k, v in prerequisites:
    
    if v not in graph:
        graph[v] = [k]
    else:
        graph[v].append(k)
        
for items in cache:
    if items not in graph:
        graph[items] = []

print(graph)
def dfs_topsort(graph):         # recursive dfs with 
    
    L = []                      # additional list for order of nodes
    color = { u : "white" for u in graph }
    found_cycle = [False]
    
    for u in graph:
        
        if color[u] == "white":
            dfs_visit(graph, u, color, L, found_cycle)
        
        if found_cycle[0]:
            break
 
    if found_cycle[0]:           # if there is a cycle, 
        L = []                   # then return an empty list  

    L.reverse() # reverse the list
    
    if numCourses > len(L):
        res = [x for x in range(numCourses) if x not in L]
        L = res + L

    
    return L                     # L contains the topological sort
 
 
def dfs_visit(graph, u, color, L, found_cycle):
    
    if found_cycle[0]:
        return
    
    color[u] = "gray"
    
    for v in graph[u]:
        
        if color[v] == "gray":
            found_cycle[0] = True
            return
        
        if color[v] == "white":
            dfs_visit(graph, v, color, L, found_cycle)
    
    color[u] = "black"      # when we're done with u,
    L.append(u)             # add u to list (reverse it later!)

dfs_topsort(graph)


# In[17]:


numCourses =  2
prerequisites = [[1,0],[0,1]]

graph = {}

cache = set([x for x in prerequisites for x in x])

for k, v in prerequisites:
    if v not in graph:
        graph[v] = [k]
    else:
        graph[v].append(k)

for items in cache:
    if items not in graph:
        graph[items] = []

def dfs_topological(graph):
    
    L = []
    color = {u:'white' for u in graph}
    found_cycle = [False]
    
    for nodes in graph:
        
        if color[nodes] == 'white':
            dfs_visit(nodes, graph, color, L, found_cycle)
        
        if found_cycle[0]:
            return "No topological Sort exist"
    
    L.reverse()
    
    if numCourses > len(L):
        L = [x for x in range(numCourses) if x not in L] + L
    
    return L

def dfs_visit(nodes, graph, color, L, found_cycle):
    
    if found_cycle[0]:
        return
    
    color[nodes] = 'gray'
    
    for v in graph[nodes]:
        
        if color[v] == 'gray':
            found_cycle[0] = True
            
        if color[v] == 'white':  
            dfs_visit(v, graph, color, L, found_cycle)
    
    color[nodes] = 'black'
    L.append(nodes)

dfs_topological(graph)


# # 4 Sum

# In[57]:


import itertools
nums = [1,0,-1,0,-2,2]
target = 0
nums.sort()

ans = []

for x, items in enumerate(nums):
    
    temp_target = target - items
    
    for x1, items1 in enumerate(nums[x + 1:], x):
        
        temp_target1 = temp_target - items1
        cache = nums[x1 + 2:]
        start, end = 0, len(cache) - 1
        
        while start < end:
            
            if cache[start] + cache[end] == temp_target1:
                
                ans.append([items, items1, cache[start], cache[end]])
                start += 1
                end -= 1
                
            elif cache[start] + cache[end] > temp_target1:
                end -= 1
            
            elif cache[start] + cache[end] < temp_target1:
                start += 1
ans.sort()
list(ans for ans,_ in itertools.groupby(ans))


# # Fizzbuzz

# In[ ]:


cache = [' Ivani' * (not i % 3) + ' is' * (not i % 4) + ' cool' * (not i % 5) for i in range(1, 100+1)]
print(*[x if y == '' else str(x) + y if y != ' is cool' else str(x) for x, y in enumerate(cache, 1)], sep = '\n')


# # Find day

# In[11]:


a = 23
s = 'Thu'
# Find day which falls after 23 days starting Thursday

cache = {'1':'Mon', '2':'Tue', '3':'Wed','4':'Thu','5':'Fri','6':'Sat','7':'Sun'}
ans = ''

if a > 7:
    temp = a % 7
    temp = temp + int([x for x, y in cache.items() if y == s][0])
    
    if temp > 7:
        temp = temp % 7
        ans = cache[str(temp)]
    
    else:
        ans = cache[str(temp)]
        
else:
    
    temp = a + int([x for x, y in cache.items() if y == s][0])

    if temp > 7:
        temp = temp % 7
        ans = cache[str(temp)]

    else:
        ans = cache[str(temp)]
ans


# # Reveal Cards in Increasing Order

# In[138]:


def swap(L):
    start = -1
    while ~start < len(L) - 1: 
        L[start],L[start - 1] = L[start - 1],L[start]
        start -= 1
    return L

deck = [4,8,11,45,12,34,19]
deck.sort()
ans = [deck.pop()]

while deck:
    
    if len(deck) == 1:
        ans.append(deck.pop())
        ans = swap(ans)
    else:
        ans = [deck.pop()] + ans
        ans = swap(ans)
ans


# In[146]:


#https://leetcode.com/problems/reveal-cards-in-increasing-order/discuss/201010/Python-easy-solution-with-detailed-explanation-(44ms-beats-100)
#Faster Solution

deck = [4,8,11,45,12,34,19]
temp = deck[:]
temp.sort(reverse=True)
ans = []

for i in range(len(temp)):
    if i <= 1:
        ans = [temp[i]] + ans
    else:
        val = ans.pop()
        ans = [temp[i]] + [val] + ans
deck

#Deque
from collections import deque

deck.sort(reverse = True)
ans = deque()
for card in deck:
    if len(ans) != 0:
        ans.appendleft(ans.pop())
    ans.appendleft(card)

list(ans)


# # 967. Numbers With Same Consecutive Differences

# In[92]:


from collections import deque

N = 6
K = 3

if N == 1: return list(range(10))

cache = deque(list(range(1,10)))
while len(str(cache[-1])) < N:
    
    number = cache.pop()
    units = number % 10
    
    if units - K >= 0:
        temp = abs(units - K)
        final_number = int(str(number) + str(temp))
        cache.appendleft(final_number)
    
    if units + K < 10:
        temp = units + K
        final_number = int(str(number) + str(temp))
        cache.appendleft(final_number)


# # 94. Binary tree in order traversal

# In[ ]:


#Recursive

class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        self.helper(res, root)
        return res
    
    def helper(self, res,  root):
        
        if root:
            self.helper(res, root.left)
            res.append(root.val)
            self.helper(res, root.right)

#Iterative

def inorderTraversal(self, root):
    ans = []
    stack = []
    
    while stack or root:
        if root:
            stack.append(root)
            root = root.left
        else:
            tmpNode = stack.pop()
            ans.append(tmpNode.val)
            root = tmpNode.right
        
    return ans


# # Maximal Square

# In[604]:


M = [["1","0","1","0","0"],
     ["1","0","1","1","1"],
     ["1","1","1","1","1"],
     ["1","0","0","1","0"]]
M = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
R = len(M)
C = len(M[0])

S = [[0 for k in range(C)] for l in range(R)]
S[0] = M[0]
for i in range(R):
    S[i][0] = M[i][0]

S


# In[606]:


best = 0
for i in range(1, R):
    for j in range(1, C):
        
        if M[i][j] == '1':
            S[i][j] = min(int(S[i][j - 1]), int(S[i - 1][j]), int(S[i - 1][j - 1])) + 1
            best = max(best, S[i][j])
        else:
            S[i][j] = 0
best


# # Fruits into Baskets

# In[4]:


#tree = [0,1,6,6,4,4,6]
#tree = [3,3,3,1,2,1,1,2,2,3,3,4]
tree = [1,2,3,2,2]

f1 = 0
f2 = [x for x, y in enumerate(tree) if y!= tree[0]][0]

pair = [tree[f2], tree[f1]]
ans = f2 - f1

for indexes, items in enumerate(tree[f2 + 1:], f2 + 1):
    
    if items in pair:
        f2 = indexes
        ans = max(ans, f2 - f1)

    else:    
        ans = max(ans, f2 - f1)
        f1 = indexes - 1
        
        while tree[f1 - 1] == tree[f2]:            
            f1 -= 1
            
        print(f1, indexes)
        f2 = indexes
        pair = [tree[f2], tree[f1]]
ans + 1


# # Available Captures for Rock

# In[60]:


L = [[".",".",".",".",".",".",".","."],
     [".",".",".",".","p",".","p","."],
     [".","p",".",".",".",".",".","."],
     [".","p","B",".","R",".","B","p"],
     [".","p",".",".",".","B",".","."],
     [".",".","p",".",".","p",".","."],
     [".",".",".",".",".",".",".","."],
     [".",".",".",".",".",".",".","."]]
res = 0

x, y = 0, 0
for i in range(len(L)):
    for j in range(len(L[0])):
        if L[i][j] == "R":
            x, y = i, j
ini1, ini2 = x, y


while x >= 0:
    
    x -= 1
    if x >= 0 and L[x][y].isupper():
        break

    elif x >= 0 and L[x][y] == "p":
        res += 1
        break
x, y = ini1, ini2

while x < 8:

    x += 1

    if x < len(L) and L[x][y].isupper():
        break

    elif x < len(L) and L[x][y] == "p":
        res += 1
        break
x, y = ini1, ini2

while y >= 0:

    y -= 1

    if y >= 0 and L[x][y].isupper():
        break

    elif y >= 0 and L[x][y] == "p":
        res += 1
        break
x, y = ini1, ini2

while y < len(L[0]):

    y += 1

    if y < len(L[0]) and L[x][y].isupper():
        break

    elif y < len(L[0]) and L[x][y] == "p":
        res += 1
        break
res


# # Merge Overlapping Intervals

# In[84]:


#intervals = [[1,3],[8,10],[15,18],[2,6]]

intervals = [[1,3],[2,6],[1,100],[15,18]]
intervals.sort(key = lambda x:x[0])

res = [intervals[0]]

for x, y in intervals[1:]:
    
    tmp1, tmp2 = res[-1][0], res[-1][1]
    
    if x <= tmp2:

        res[-1] = [tmp1, max(y, tmp2)]
    
    else:
        res.append([x, y])
res


# In[ ]:


def merge(self, intervals: List[Interval]) -> List[Interval]:

    if not intervals: return []

    intervals.sort(key = lambda x:x.start)

    res = [[intervals[0].start, intervals[0].end]]

    for x in intervals[1:]:

        tmp1, tmp2 = res[-1][0], res[-1][1]
        if x.start <= tmp2:

            res[-1] = [tmp1, max(x.end, tmp2)]

        else:
            res.append([x.start, x.end])
    return res


# # Game of Life

# In[131]:


board = [
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]


# Any live cell with fewer than two live neighbors dies, as if caused by under-population.
# Any live cell with two or three live neighbors lives on to the next generation.
# Any live cell with more than three live neighbors dies, as if by over-population..
# Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.

def dfs(i, j):

    L = [
          [0,1,0],
          [0,0,1],
          [1,1,1],
          [0,0,0]
        ]

    if i < 0 or i >= len(L) or j < 0 or j >= len(L[0]):
        return 0

    #print(list(map(dfs, (i + 1, i - 1, i, i, i - 1, i + 1, i + 1, i - 1), (j, j, j + 1, j - 1, j - 1, j + 1, j - 1, j + 1))))

    return L[i][j]

for i in range(len(L)):
    for j in range(len(L[0])):
        res = list(map(dfs, (i + 1, i - 1, i, i, i - 1, i + 1, i + 1, i - 1), (j, j, j + 1, j - 1, j - 1, j + 1, j - 1, j + 1)))
        temp = sum(res)

        if L[i][j] == 1:
            if temp < 2 or temp > 3:
                L[i][j] = 0

        elif L[i][j] == 0:
            if temp == 3:
                L[i][j] = 1
L


# In[133]:


def dfs(i, j):

    L = [
          [0,1,0],
          [0,0,1],
          [1,1,1],
          [0,0,0]
        ]

    if i < 0 or i >= len(L) or j < 0 or j >= len(L[0]):
        return 0

    #print(list(map(dfs, (i + 1, i - 1, i, i, i - 1, i + 1, i + 1, i - 1), (j, j, j + 1, j - 1, j - 1, j + 1, j - 1, j + 1))))

    return L[i][j]

res = []
for i in range(len(L)):
    for j in range(len(L[0])):
        res.append(list(map(dfs, (i + 1, i - 1, i, i, i - 1, i + 1, i + 1, i - 1), (j, j, j + 1, j - 1, j - 1, j + 1, j - 1, j + 1))))

for i in range(len(L)):
    for j in range(len(L[0])):
        
        temp = sum(res[0])
        
        if L[i][j] == 1:
            if temp < 2 or temp > 3:
                L[i][j] = 0

        elif L[i][j] == 0:
            if temp == 3:
                L[i][j] = 1
        del(res[0])
L


# # Word Break

# In[213]:


s = 'cars'
wordDict = ['car','ca', 'rs']
dp = [0]*(len(s) + 1)
dp[0] = True

for i in range(len(s)):
    for j in range(i, len(s)):
        if dp[i] and s[i:j + 1] in wordDict:
            dp[j + 1] = True
dp


# # Trapping Rain Water

# In[16]:


height = [1,0,2,1,0,1,3,2,1,2,1]
curr, ans = height[0],0
curr_max = height[0]

for items in height[1:]:
    
    if items < curr_max:
        
        ans += curr - items
        curr = items
        
    elif items > curr_max:
        


# # Search a 2D Matrix

# In[224]:


matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]

def binary(L, target):
    
    start = 0
    end = len(L) - 1

    while start <= end:

        mid = start + (end - start) // 2

        if L[mid] == target:
            return True
        
        elif target > L[mid]:
            start = mid + 1

        else:
            end = mid - 1
    return False

found = False
for items in range(len(matrix)):
    
    if matrix[items][0] > target:
        break
    
    elif matrix[items][0] <= target <= matrix[items][-1]:
        
        if binary(matrix[items], target):
            found = not found
            break
#return Found


# # Search in 2D matrix II

# In[234]:


if not matrix or matrix == [[]]: return False

cache = set([target])

for items in range(len(matrix)):

    if matrix[items][0] > target and matrix[items][-1] > target:
        break

    elif matrix[items][0] > target:
        continue

    elif matrix[items][0] <= target <= matrix[items][-1]:
        if cache & set(matrix[items]):
            return True

return False


# # Spiral Matrix

# In[18]:


class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        
        if not matrix: return []
        if len(matrix[0]) == 1: return [x for x in matrix for x in x]
        
        start, window, end = 0, len(matrix[0]), len(matrix)
        res, top, right, bottom, left = [], [], [], [], []
        total_elements = len(matrix) * len(matrix[0]) 
        
        while len(res) != total_elements:
            
            res += self.recur(matrix, start, end, window, top, right, bottom, left,res )
            #res += list(map(self.recur, start, end, window, top, right, bottom, left))
            top, right, bottom, left = [], [], [], []
            start += 1
            end -= 1
            window -= 1
            
        return res
            
    def recur(self, matrix, start, end, window, top, right, bottom, left, res):
                
        for items in range(start, end):
            
            if items == start:
                top += matrix[items][start:window]

            elif items != start and items != end - 1:
                left += matrix[items][start],
                right += matrix[items][window - 1],

            elif items == end - 1:
                bottom += matrix[items][start:window]
                
        bottom.reverse()
        left.reverse()
        
        return top + right + bottom + left


# In[42]:


matrix = [
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
#if not matrix: return []
res = []
while matrix:
    res += matrix.pop(0)
    matrix = list(zip(*matrix))[::-1]

res


# # Find common characters

# In[401]:


from collections import Counter
class Solution:
    def commonChars(self, A: List[str]) -> List[str]:
        
        if not A: return []
        
        x = Counter(A[0])
        
        for items in A[1:]:
            x = x & Counter(items)
            
        res = [[x] * y for x, y in x.items()]
        return [x for x in res for x in x]


# # Check if word is valid

# In[420]:


class Solution:
    def isValid(self, S: str) -> bool:
        
        if S == "abc": return True
        while S:
            
            if "abc" not in S:
                break
                
            S = S.split("abc")
            S = ''.join([x for x in S if x != ''])
        return True if S == '' or S == "abc" else False


# # Minimum Cost to merge stones

# In[204]:


L = [4,6,4,7,5]
K = 2
curr = 0
while L:
    
    if len(L) == K:
        curr += sum(L)
        break
        
    cache = [sum(L[i:K + i]) for i in range(0,len(L) - K + 1)]    
    
    if cache == []:
        break
        #return -1
    curr += min(cache)
    res = cache.index(min(cache))
    temp = min(cache)
    L[res : res + K] = [temp]

curr


# In[208]:


stones = [4, 6, 4, 7, 5]
K = 2
import functools
p = functools.reduce(lambda s,x: s+[s[-1]+x],stones,[0])
@functools.lru_cache(None)
def dfs(i,j,m):
    if (j-i+1-m)%(K-1): return math.inf
    if i==j: return 0 if m==1 else math.inf
    if m==1: return dfs(i,j,K)+p[j+1]-p[i]
    return min(dfs(i,k,1)+dfs(k+1,j,m-1) for k in range(i,j))
res = dfs(0,len(stones)-1,1)
res


# # Maximum Consecutive Ones

# In[62]:


from collections import deque, Counter
#A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1]
A = [1,1,1,0,0,0,1,1,1,1,0]
K = 1
res = deque([])
ans = len(res)
start = 0
cache = 0
while start < len(A):
    
    res += A[start],
#    cache = Counter(res)[0]
    if A[start] == 0:
        cache += 1
    
    if cache > K:
        while cache > K:
            temp = res.popleft()
            if temp == 0:  
                cache -= 1

        
    #res += A[start],        
    start += 1
    ans = max(ans, len(res))
ans


# # Next Permutation

# In[115]:


# https://www.nayuki.io/page/next-lexicographical-permutation-algorithm
nums = [0,1,2,5,3,3,0]
curr = nums[-1]
pivot = -1
for items in nums[-2::-1]:
    if items >= curr:
        pivot -= 1
        curr = items
    else:
        break
if pivot == - len(nums):print('break')
j = len(nums) - 1
while nums[j] <= nums[pivot - 1]:
    j -= 1
nums[j], nums[pivot - 1] = nums[pivot - 1], nums[j]
nums[pivot:] = nums[pivot:][::-1]
nums


# # Minimum Path Sum

# In[125]:


grid = [
  [1,3,1],
  [1,1,1],
  [4,2,2]
]

for i in range(len(grid)):
    for j in range(len(grid[0])):
        
        if i == 0 and j == 0:
            pass
        
        elif i == 0:
            grid[i][j] += grid[i][j - 1]
        
        elif j == 0 and i != 0:
            print(i, j)
            grid[i][j] += grid[i - 1][j]
        
        else:
            grid[i][j] += min(grid[i - 1][j], grid[i][j - 1])
grid[-1][-1]


# # Unique Paths II

# In[163]:


class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        
        if not obstacleGrid or obstacleGrid == [[]]: return 0
        
        if len(obstacleGrid[0]) == 1:
            cache = [x for x in obstacleGrid for x in x]
            if 1 in cache:
                return 0
            else:
                return 1
        
        if obstacleGrid[0][0] == 1 or obstacleGrid[-1][-1]: return 0
            
        grid = obstacleGrid[:]
        obstacle = [(i, j) for i in range(len(grid)) for j in  
                    range(len(grid[0])) if grid[i][j] == 1]

        for x, y in obstacle:
            grid[x][y] = 'Obstacle'

        for i in range(1, len(grid[0])):

            if grid[0][i] != 'Obstacle':
                grid[0][i] = 1
            else:
                break

        for i in range(1, len(grid)):

            if grid[i][0] != 'Obstacle':
                grid[i][0] = 1
            else:
                break


        for i in range(1, len(grid)):
            for j in range(1, len(grid[0])):

                if grid[i][j] == 'Obstacle':
                    pass

                elif grid[i - 1][j] == 'Obstacle' and grid[i][j - 1] != 'Obstacle':
                    grid[i][j] = grid[i][j - 1]

                elif grid[i][j - 1] == 'Obstacle' and grid[i - 1][j] != 'Obstacle':
                    grid[i][j] = grid[i - 1][j]

                elif grid[i][j - 1] == 'Obstacle' and grid[i - 1][j] == 'Obstacle':
                    grid[i][j] = 0

                else:
                    grid[i][j] = sum([grid[i - 1][j], grid[i][j - 1]])
                    
        return grid[-1][-1] if grid[-1][-1] != 'Obstacle' else 0


# # 0 - 1 Knapsack

# In[210]:


def knapsack(items, maxweight):
    # Create an (N+1) by (W+1) 2-d list to contain the running values
    # which are to be filled by the dynamic programming routine.
    #
    # There are N+1 rows because we need to account for the possibility
    # of choosing from 0 up to and including N possible items.
    # There are W+1 columns because we need to account for possible
    # "running capacities" from 0 up to and including the maximum weight W.
    bestvalues = [[0] * (maxweight + 1)
                  for i in range(len(items) + 1)]
    # Enumerate through the items and fill in the best-value table
    for i, (value, weight) in enumerate(items, 1):
        # Increment i, because the first row (0) is the case where no items
        # are chosen, and is already initialized as 0, so we're skipping it
        #i += 1
        for capacity in range(maxweight + 1):
            # Handle the case where the weight of the current item is greater
            # than the "running capacity" - we can't add it to the knapsack
            if weight > capacity:
                bestvalues[i][capacity] = bestvalues[i - 1][capacity]
            else:
                # Otherwise, we must choose between two possible candidate values:
                # 1) the value of "running capacity" as it stands with the last item
                #    that was computed; if this is larger, then we skip the current item
                # 2) the value of the current item plus the value of a previously computed
                #    set of items, constrained by the amount of capacity that would be left
                #    in the knapsack (running capacity - item's weight)
                candidate1 = bestvalues[i - 1][capacity]
                candidate2 = bestvalues[i - 1][capacity - weight] + value

                # Just take the maximum of the two candidates; by doing this, we are
                # in effect "setting in stone" the best value so far for a particular
                # prefix of the items, and for a particular "prefix" of knapsack capacities
                bestvalues[i][capacity] = max(candidate1, candidate2)

    # Reconstruction
    # Iterate through the values table, and check
    # to see which of the two candidates were chosen. We can do this by simply
    # checking if the value is the same as the value of the previous row. If so, then
    # we say that the item was not included in the knapsack (this is how we arbitrarily
    # break ties) and simply move the pointer to the previous row. Otherwise, we add
    # the item to the reconstruction list and subtract the item's weight from the
    # remaining capacity of the knapsack. Once we reach row 0, we're done
    reconstruction = []
    i = len(items)
    j = maxweight
    while i > 0:
        if bestvalues[i][j] != bestvalues[i - 1][j]:
            reconstruction.append(items[i - 1])
            j -= items[i - 1][1]
        i -= 1

    # Reverse the reconstruction list, so that it is presented
    # in the order that it was given
    reconstruction.reverse()

    # Return the best value, and the reconstruction list
    return bestvalues[len(items)][maxweight], reconstruction
#knapsack([[60, 49], [10, 5], [12,6]], 50)


# # Maximize Sum Of Array After K Negations

# In[15]:


A = [1,1,1]
K = 3
for i in range(K):
    
    A[A.index(min(A))] = - A[A.index(min(A))]
A


# # Clumsy Factorial

# In[1]:


N = 4
cache = [0]* N *2

for x, y in zip(range(N, 0, -1), range(0, N * 2, 2)):
    
    cache[y] = x

cache.pop()
operations = ['*', '//', '+', '-']

temp = 0

for i in range(1, len(cache), 2):
    
    if temp != 3:
        cache[i] = operations[temp]
        temp += 1
    
    else:
        cache[i] = operations[temp]
        temp = 0
        
cache = ''.join(map(str, cache))
eval(cache)


# # Minimum Domino Rotations For Equal Row

# In[2]:


from collections import Counter

A = [1,2,2,2]
B = [2,2,2,5]

s = [A[0], B[0]]

for items in zip(A[1:], B[1:]):
    
    s = set(s) & set(list(items))
s


# # Triangle

# In[235]:


triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]

for items in range(len(triangle) - 2, -1, -1):
    
    for idx in range(len(triangle[items])):
        
        temp = triangle[items][idx]
        triangle[items][idx] = min(temp + triangle[items + 1][idx], temp + triangle[items + 1][idx + 1])

triangle[0][0]


# # Subdomain Visit Count

# In[265]:


cpdomains = ["9001 discuss.leetcode.com", "9001 discuss.leetcode.com"]

cache = {}

for items in cpdomains:
    
    temp= items.split(' ')
    temp[0] = int(temp[0])
    
    if temp[-1] in cache:
        cache[temp[-1]] += temp[0]
    else:
        cache[temp[-1]] = temp[0]
    
com = 0

for items in cpdomains:
    
    temp = items.split(' ')
    temp[-1] = temp[-1].split('.')
    temp[0] = int(temp[0])
    temp[-1] = [x + '.' + temp[-1][-1] for x in temp[-1] if x != temp[-1][-1]] + [temp[-1][-1]]
    
    for items in temp[-1][1:]:
        
        if items not in cache:
            cache[items] = temp[0]
        else:
            cache[items] += temp[0]
[' '.join(list((str(v), k))) for k, v in cache.items()]


# # Coin Change

# In[2]:


coins = [2]
amount = 3

need = [amount + 1] + [amount + 1] * amount
for c in coins:
    for a in range(c, amount+1):
        need[a] = min(need[a], need[a - c] + 1)

need


# # Meeting rooms II

# In[3]:


from collections import defaultdict
#intervals = [[0,30],[5,10],[15,20], [1, 8], [1, 2]]
intervals = [[7,10],[2,4], [1, 2], [1, 8], [1, 6], [5, 6]]
intervals = [[7,10],[2,4], [1, 2], [1, 8], [1, 6], [5, 6], [6, 7], [7, 8], [5,8]]

intervals.sort(key = lambda x:x[0])

track = {1:[[intervals[0][0], intervals[0][-1]]]}
curr_room = 1

for x, y in intervals[1:]:
    
    cache = [i for i, j in track.items() if x >= j[-1][-1]]
    if cache == []:
        curr_room += 1
        track[curr_room] = [[x, y]]
    
    else:
        track[cache[-1]].append([x, y])
curr_room


# #  Subarray Product Less Than K

# In[229]:


from functools import reduce
import operaTtor
import math

nums = [1]*10000
nums = [10, 5, 2, 6]
k = 100

start, end, curr, ans = 0, 1, 1, 0

while start < len(nums) and end < len(nums):
    
    if nums[start] < k:
        ans += 1
        curr *= nums[start]

    if nums[start] < k and nums[end] < k:

        curr *= nums[end]
        while curr < k and end < len(nums):
            ans += 1
            end += 1
            if end < len(nums):
                curr *= nums[end]

    start += 1
    end = start + 1
    curr = 1


    ans + 1 if nums[-1] < k else ans


# In[281]:


nums = [1]*3
k = 2

start = 0
ans = 0
curr = 1

for idx, items in enumerate(nums):
    
    curr *= items
    
    while curr >= k:
        curr /= nums[start]
        start += 1
        
    print(idx, start)
    ans += idx - start + 1

ans


# # Subarray Sum Equals K

# In[277]:


nums = [1]*3

k = 2
ans = 0
cache = (nums[i:j + 1] for i in range(len(nums)) for j in range(i, len(nums)))
cache = (sum(i) for i in cache)

for items in cache:
    if items == k:
        ans += 1
ans


# # Complement of Base 10 Integer

# In[306]:


N = 0
temp = {'1':'0', '0':'1'}
cache = list("{0:b}".format(N))
int(''.join([temp[x] for x in cache]), 2)


# # Numbers With 1 Repeated Digit

# In[18]:


N = 1000
cache = map(Counter, map(str,(set(range(N + 1)))))
ans = 0
for item in cache:
    temp = next((x for x, y in item.items() if y > 1), '')
    if temp != '':
        ans += 1 
ans

# XOR
for items in range(10, N + 1):
    cache = str(items)


# # Capacity To Ship Packages Within D Days

# In[ ]:


weights = [1,2,3,4,5,6,7,8,9,10]
D = 5
start = 0
res = []
while D != 0:
    
    temp = []
    


# # Pairs of Songs With Total Durations Divisible by 60

# In[63]:


import collections
time = [30,20,150,100,40]
res = 0
cache = collections.Counter()
for t in time:
    res += cache[-t % 60]
    print(cache)
    cache[t % 60] += 1
cache


# # Google 1st Interview

# In[14]:


cache = {chr(i + 97): str(i) for i in range(26)}
S2 = 'ccb'

cache = {chr(i + 97): i for i in range(26)}

if not S2: return 0

curr_position = 0
answer = 0
prev_char = ''

for idx, items in enumerate(S2):
    

    if items == prev_char:
        pass
    
    else:
        prev_char = items
        curr_position = idx
        answer += abs(curr_position - cache[items])
    
answer


# # Google 2nd

# In[ ]:


A = [1,2,3,4,5,6]
levels = {}
for i in range(len(A)):
    
    if i*2 + 1 < len(A) and i*2 + 2 < len(A):
    
        levels[i + 1] = [A[i*2 + 1], A[i*2 + 2]]
    
    elif i*2 + 1 < len(A) and i*2 + 2 >= len(A):
        levels[i + 1] = [A[i*2 + 1]]
    
    elif i*2 + 1 >= len(A):
        break
        
levels


# 
# # Binary String With Substrings Representing 1 To N

# In[5]:


from collections import Counter

S = "0110"
N = 4


for i in range(1, N + 1):
    
    temp = "{0:b}".format(i)
    if temp in S:
        pass
    else:
        print(False)
print(True)


# # Best sightseeing pair

# In[27]:


A = [8,1,5,2,6]
start = A[0] - 1 
res = 0
for j in range(1, len(A)):

    res = max(res, start + A[j])
    start = max(start, A[j])
    start -= 1

res


# # Smallest Integer Divisible by K

# In[20]:





# # Binary Prefix divisible by 5

# In[34]:


A = [0,1,1,1,1,1]
res = ''
ans = []
for items in A:
    res += str(items)
    if not int(res, 2) % 5:
        ans += True,
        
    else:
        ans += False,
ans


# # Convert to Base -2

# In[37]:


int("111", 2)


# # Remove Outermost Parentheses

# In[26]:




