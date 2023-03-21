# 自用板子（cpp）

[TOC]

## 加速框架

```cpp
#include <bits/stdc++.h>

#define int long long

#pragma GCC optimize(2) //不到万不得已别用

typedef long long LL;

using namespace std;

const int N = 1e5 + 10;

signed main(){
    ios::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
	return 0;
}
```

## 基础算法

### 排序

#### 快速排序

```cpp
void quick_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    quick_sort(q, l, j), quick_sort(q, j + 1, r);

}
```

#### 归并排序

```cpp
void merge_sort(int q[], int l, int r)
{
    if (l >= r) return;

    int mid = l + r >> 1;
    merge_sort(q, l, mid);
    merge_sort(q, mid + 1, r);
    
    int k = 0, i = l, j = mid + 1;
    while (i <= mid && j <= r)
        if (q[i] <= q[j]) tmp[k ++ ] = q[i ++ ];
        else tmp[k ++ ] = q[j ++ ];
    
    while (i <= mid) tmp[k ++ ] = q[i ++ ];
    while (j <= r) tmp[k ++ ] = q[j ++ ];
    
    for (i = l, j = 0; i <= r; i ++, j ++ ) q[i] = tmp[j];

}
```

### 二分

#### 整数二分

```cpp
bool check(int x) {/* ... */} // 检查x是否满足某种性质

// 区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = l + r + 1 >> 1;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return l;
}
```

#### 浮点数二分

```cpp
bool check(double x) {/* ... */} // 检查x是否满足某种性质

double bsearch_3(double l, double r)
{
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps)
    {
        double mid = (l + r) / 2;
        if (check(mid)) r = mid;
        else l = mid;
    }
    return l;
}
```

### 高精度

#### 高精度加法

```cpp
// C = A + B, A >= 0, B >= 0
vector<int> add(vector<int> &A, vector<int> &B)
{
    if (A.size() < B.size()) return add(B, A);

    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ )
    {
        t += A[i];
        if (i < B.size()) t += B[i];
        C.push_back(t % 10);
        t /= 10;
    }
    
    if (t) C.push_back(t);
    return C;

}
```

#### 高精度减法

```cpp
// C = A - B, 满足A >= B, A >= 0, B >= 0
vector<int> sub(vector<int> &A, vector<int> &B)
{
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ )
    {
        t = A[i] - t;
        if (i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if (t < 0) t = 1;
        else t = 0;
    }

    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;

}
```

#### 高精度乘低精度

```cpp
// C = A * b, A >= 0, b >= 0
vector<int> mul(vector<int> &A, int b)
{
    vector<int> C;

    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }
    
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    
    return C;

}
```

#### 高精度除以低精度

```cpp
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

#### 高精度乘高精度

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 1e5;
const int M = 1e8;
int a[N], b[N], c[M];
int main(){
    char a1[N], b1[N];
    int lena, lenb, lenc, jw = 0;
    cin >> a1 >> b1;
    lena = strlen(a1);
    lenb = strlen(b1);
    lenc = lena + lenb;
    for (int i = 0; i < lena; i++) a[i] = a1[lena - i - 1] - '0';
    for (int i = 0; i < lenb; i++) b[i] = b1[lenb - i - 1] - '0';
    for (int i = 0; i < lena; i++){
        for (int j = 0; j < lenb; j++){
            c[i + j] += a[i] * b[j] + jw;
            jw = c[i + j] / 10;
            c[i + j] %= 10;
        }
        c[i + lenb] = jw;
    }
    for (int i = lenc - 1; i >= 0; i--){
        if (0 == c[i] && lenc > 1) lenc--;
        else break;
    }
    for (int i = lenc - 1; i >= 0; i--) cout << c[i];
    return 0;
}
```

### 离散化



## 数论

### 快速幂

```cpp
int qpow(int a, int b, int c){
	int res = 1;
    while(b){
        if(b & 1) res = res * a % c; //不可写成res *= a % c
        b >>= 1;
        a = a * a % c;
    }
    return res;
}
```

### 逆元

#### 费马小定理求逆元

费马小定理(Fermat's little theorem)是[数论](https://baike.baidu.com/item/数论/3700?fromModule=lemma_inlink)中的一个重要[定理](https://baike.baidu.com/item/定理/9488549?fromModule=lemma_inlink)，在1636年提出。如果p是一个[质数](https://baike.baidu.com/item/质数/263515?fromModule=lemma_inlink)，而[整数](https://baike.baidu.com/item/整数/1293937?fromModule=lemma_inlink)a不是p的倍数，则有

$a^{p-1} \equiv 1 (mod \ p)$

对上式变形得

$a^{p-2}a\equiv1(mod \ p)$

则a mod p的逆元是$a^{p-2}$

接下来通过`qpow(a, p - 2, p)`即可求出a mod p的逆元

## 图论

## 数据结构

### 链表

#### 单链表（静态）

```cpp
//head是头指针，e存权值，ne存下一个节点，idx分配节点
int head, e[N], ne[N], idx;

void init(){
    head = -1; //链表中最后一个节点的ne始终是-1
    idx = 0;
}

void add_to_head(int x){ //在头结点前建立节点
    e[idx] = x;
    ne[idx] = head; 
    head = idx ++;
}

void add_to_k(int k, int x){
    e[idx] = x;
    ne[idx] = ne[k];
    ne[k] = idx ++;
}

void remove(int k){
    ne[k] = ne[ne[k]];
}
```

#### 动态链表

例题1.

实现一个单链表，链表初始为空，支持三种操作：

1.  向链表头插入一个数；
2.  删除第 $ k $ 个插入的数后面的数；
3.  在第 $ k $ 个插入的数后插入一个数。

现在要对该链表进行 $ M $ 次操作，进行完所有操作后，从头到尾输出整个链表。

**注意**:题目中第 $ k $ 个插入的数并不是指当前链表的第 $ k $ 个数。例如操作过程中一共插入了 $ n $ 个数，则按照插入的时间顺序，这 $ n $ 个数依次为：第 $ 1 $ 个插入的数，第 $ 2 $ 个插入的数，…第 $ n $ 个插入的数。

输入格式

第一行包含整数 $ M $，表示操作次数。

接下来 $ M $ 行，每行包含一个操作命令，操作命令可能为以下几种：

1.  `H x`，表示向链表头插入一个数 $ x $。
2.  `D k`，表示删除第 $ k $ 个插入的数后面的数（当 $ k $ 为 $ 0 $ 时，表示删除头结点）。
3.  `I k x`，表示在第 $ k $ 个插入的数后面插入一个数 $ x $（此操作中 $ k $ 均大于 $ 0 $）。

输出格式

共一行，将整个链表从头到尾输出。

数据范围

$ 1 \le M \le 100000 $  
所有操作保证合法。

输入样例：

```
10
H 9
I 1 1
D 1
D 0
H 6
I 3 6
I 4 5
I 4 5
I 3 4
D 6
```

输出样例：

```
6 4 6 5
```

```cpp
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

struct Node{
    int num;
    struct Node *next;
};

struct Node * push_front(int x, struct Node *h){
    if(h == NULL){
        h = malloc(sizeof(struct Node));
        h -> num = x;
        h -> next = NULL;
        return h;
    }
    else{
        struct Node *new = malloc(sizeof(struct Node));
        new->num = x;
        new->next = h;
        return new;
    }
}

struct Node * push_back(int x, struct Node *h){
    if(h == NULL){
        h = malloc(sizeof(struct Node));
        h -> num = x;
        h -> next = NULL;
        return h;
    }
    else {
        struct Node *new = malloc(sizeof(struct Node));
        new->num = x;
        new->next = NULL;
        struct Node *p = h;
        while(p -> next != NULL){
            p = p -> next;
        }
        p -> next = new;
        return h;
    }
}

void print(struct Node *h){
    struct Node *p = h;
    while(p != NULL) {
        printf("%d ", p->num);
        p = p -> next;
    }
}

void connect(struct Node *h){
    struct Node *p = h;
    while(p -> next != NULL){
        p = p -> next;
    }
    p -> next = h;
}

void recycle(struct Node *p){
    struct Node *cpy = p;
    p = p -> next;
    while(p != NULL){
        struct Node *tmp = p;
        p = p -> next;
        free(tmp);
    }
    cpy -> next = p;
}

void add(int x, int y, struct Node *h){
    struct Node *p = h;
    while(p != NULL){
        if(p -> num == x){
            struct Node *new = malloc(sizeof(struct Node));
            new -> num = y;
            new -> next = p -> next;
            p -> next = new;
            return;
        }
        p = p -> next;
    }
}

void change(int x, int y, struct Node *h){
    struct Node *p = h;
    while(p != NULL){
        if(p -> num == x){
            p -> num = y;
            return;
        }
        p = p -> next;
    }
}

void delete(struct Node *p){
    p -> next = p -> next -> next;
    free(p -> next);
}

int main()
{
    struct Node *head = NULL;
    for(int i = 1; i <= 10; i ++){
        head = push_back(i, head);
    }
    add(6, 233, head);
    print(head);
    return 0;
}
```

#### 双链表

```cpp
int r[N], l[N], e[N], idx;

//1节点和0节点相当于变相的head指针，没有e值
void init(){
    r[0] = 1;
    l[1] = 0;
    idx = 2;
}

//在k节点的右边插入值为x的节点
void insert(int k, int x){
    e[idx] = x;
    l[idx] = k;
    r[idx] = r[k];
    l[r[k]] = idx; //必须先做这个，否则后面的r[k]会被修改掉
    r[k] = idx;
    idx ++;
}

void remove(int k){
    r[l[k]] = r[k];
    l[r[k]] = l[k];
}
```



### 队列

#### 一般队列

```cpp
int hh = 0, tt = -1;
int q[N];

void push(int x){
	q[++tt] = x;
}

void pop(){
	hh++;
}

int front(){
    return q[hh];
}
```

#### STL队列

```cpp
#include <queue>

queue<int> q;
```

queue 和 stack 有一些成员函数相似，但在一些情况下，工作方式有些不同：

- front()：返回 queue 中第一个元素的引用。如果 queue 是常量，就返回一个常引用；如果 queue 为空，返回值是未定义的。
- back()：返回 queue 中最后一个元素的引用。如果 queue 是常量，就返回一个常引用；如果 queue 为空，返回值是未定义的。
- push(const T& obj)：在 queue 的尾部添加一个元素的副本。这是通过调用底层容器的成员函数 push_back() 来完成的。
- push(T&& obj)：以移动的方式在 queue 的尾部添加元素。这是通过调用底层容器的具有右值引用参数的成员函数 push_back() 来完成的。
- pop()：删除 queue 中的第一个元素。
- size()：返回 queue 中元素的个数。
- empty()：如果 queue 中没有元素的话，返回 true。
- emplace()：用传给 emplace() 的参数调用 T 的构造函数，在 queue 的尾部生成对象。
- swap(queue<T> &other_q)：将当前 queue 中的元素和参数 queue 中的元素交换。它们需要包含相同类型的元素。也可以调用全局函数模板 swap() 来完成同样的操作。

#### 循环队列

主要在bfs及spfa算法中使用

```cpp
void bfs(int x){
    int q[N];
    int hh = 0, tt = 0;
    q[tt++] = x;
    st[x] = true;
    while(hh != tt){
        int t = q[hh++];
        if(hh == N) hh = 0;
        for(){
            
        }
    }
}
```

### 树状数组和线段树

#### 树状数组

“树状数组的下标从1开始,不可以从0开始,因为lowbit(0)=0时会出现死循环”

```cpp
int n, m;
int a[N];
int tr[N];

int lowbit(int x){ //lowbit这个函数的功能就是求某一个数的二进制表示中最低的一位1，举个例子，
                    //x = 6，它的二进制为110，那么lowbit(x)就返回2，因为最后一位1表示2
    return x & -x;
}

void add(int x, int v){
    for(int i = x; i <= n; i += lowbit(i)) tr[i] += v;
}

int query(int x){ //求1~x的和
    int res = 0;
    for(int i = x; i; i -= lowbit(i)) res += tr[i];
    return res;
}
```

#### 线段树

```cpp
const int N = 1e6 + 10;
int n, m;
int w[N];
struct Node{
    int l, r;
    int sum;
}tr[N * 4];

void pushup(int u){
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}

void build(int u, int l, int r){
    if(l == r) tr[u] = {l, r, w[r]};
    else{
        tr[u] = {l, r};
        int mid = l + r >> 1;
        build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
        pushup(u);
    }
}

int query(int u, int l, int r){
    if(tr[u].l >= l && tr[u].r <= r) return tr[u].sum;
    int mid = tr[u].l + tr[u].r >> 1;
    int sum = 0;
    if(l <= mid) sum = query(u << 1, l, r);
    if(r > mid) sum += query(u << 1 | 1, l, r);
    return sum;
}

void modify(int u, int x, int v){
    if(tr[u].l == tr[u].r) tr[u].sum += v;
    else{
        int mid = tr[u].l + tr[u].r >> 1;
        if(x <= mid) modify(u << 1, x, v);
        else modify(u << 1 | 1, x, v);
        pushup(u);
    }
}
```

#### 线段树类写法

例题1

给定 $ n $ 个数组成的一个数列，规定有两种操作，一是修改某个元素，二是求子数列 $ [a,b] $ 的连续和。

输入格式

第一行包含两个整数 $ n $ 和 $ m $，分别表示数的个数和操作次数。

第二行包含 $ n $ 个整数，表示完整数列。

接下来 $ m $ 行，每行包含三个整数 $ k,a,b $ （$ k = 0 $，表示求子数列$ [a,b] $的和；$ k=1 $，表示第 $ a $ 个数加 $ b $）。

数列从 $ 1 $ 开始计数。

输出格式

输出若干行数字，表示 $ k = 0 $ 时，对应的子数列 $ [a,b] $ 的连续和。

数据范围

$ 1 \le n \le 100000 $,  
$ 1 \le m \le 100000 $，  
$ 1 \le a \le b \le n $,  
数据保证在任何时候，数列中所有元素之和均在 int 范围内。

输入样例：

```
10 5
1 2 3 4 5 6 7 8 9 10
1 1 5
0 1 3
0 4 8
1 7 5
0 4 8
```

输出样例：

```
11
30
35
```

代码：

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 1e6 + 10;

int n, m;
int w[N];

class SegTree{
private:
    struct Node {
        int l;
        int r;
        int sum;
    }*tr;
public:
    void pushup(int u){
        tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
    }
    void build(int u, int l, int r){
        if(l == r) tr[u] = {l, r, w[r]};
        else{
            tr[u] = {l, r};
            int mid = l + r >> 1;
            build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
            pushup(u);
        }
    }
    SegTree(int l, int r, int length){
        tr = new Node[length * 4];
        build(1, l, r);
    }
    int query(int u, int l, int r){
        if(tr[u].l >= l && tr[u].r <= r) return tr[u].sum;
        int mid = tr[u].l + tr[u].r >> 1;
        int sum = 0;
        if(l <= mid) sum += query(u << 1, l, r);
        if(r > mid) sum += query(u << 1 | 1, l, r);
        return sum;
    }
    void modify(int u, int x, int v){
        if(tr[u].l == tr[u].r) tr[u].sum += v;
        else{
            int mid = tr[u].l + tr[u].r >> 1;
            if(x <= mid) modify(u << 1, x, v);
            else modify(u << 1 | 1, x, v);
            pushup(u);
        }
    }
    ~SegTree(){
        delete[] tr;
    }
};

int main() {
    scanf("%d%d", &n, &m);
    for(int i = 1; i <= n; i++) scanf("%d", &w[i]);
    SegTree segtree(1, n, n);
    int k, a, b;
    while(m --){
        scanf("%d%d%d", &k, &a, &b);
        if(k == 0) printf("%d\n", segtree.query(1, a, b));
        else segtree.modify(1, a, b);
    }
    return 0;
}
```

### 并查集

```cpp
//p[i]存储i节点的祖宗节点，d[i]存储i节点到祖宗节点的距离
int p[N], d[N];
//初始化
void init(){
    for(int i = 0; i <= n; i ++){
        p[i] = i;
        d[i] = 0;
    }
}
//找祖宗节点，同时更新d数组
int find(int x){
    if(x != p[x]){
        int t = find(p[x]);
        //x到祖宗的距离=x到p[x]的距离+p[x]到祖宗的距离
        //这里的距离更新仅仅是更新的d[x]正确（找祖先）的情况下，在做并差集合并时还得手动更新
        d[x] = d[x] + d[p[x]];
        p[x] = t;
    }
    return p[x];
}
int find(int x){
    if(x != p[x]) p[x] = find(p[x]);
    return p[x];
}
```

#### 例题1.食物链

动物王国中有三类动物 $ A,B,C $，这三类动物的食物链构成了有趣的环形。

$ A $ 吃 $ B $，$ B $ 吃 $ C $，$ C $ 吃 $ A $。

现有 $ N $ 个动物，以 $ 1 \sim N $ 编号。

每个动物都是 $ A,B,C $ 中的一种，但是我们并不知道它到底是哪一种。

有人用两种说法对这 $ N $ 个动物所构成的食物链关系进行描述：

第一种说法是 `1 X Y`，表示 $ X $ 和 $ Y $ 是同类。

第二种说法是 `2 X Y`，表示 $ X $ 吃 $ Y $。

此人对 $ N $ 个动物，用上述两种说法，一句接一句地说出 $ K $ 句话，这 $ K $ 句话有的是真的，有的是假的。

当一句话满足下列三条之一时，这句话就是假话，否则就是真话。

1.  当前的话与前面的某些真的话冲突，就是假话；
2.  当前的话中 $ X $ 或 $ Y $ 比 $ N $ 大，就是假话；
3.  当前的话表示 $ X $ 吃 $ X $，就是假话。

你的任务是根据给定的 $ N $ 和 $ K $ 句话，输出假话的总数。

输入格式

第一行是两个整数 $ N $ 和 $ K $，以一个空格分隔。

以下 $ K $ 行每行是三个正整数 $ D，X，Y $，两数之间用一个空格隔开，其中 $ D $ 表示说法的种类。

若 $ D=1 $，则表示 $ X $ 和 $ Y $ 是同类。

若 $ D=2 $，则表示 $ X $ 吃 $ Y $。

输出格式

只有一个整数，表示假话的数目。

数据范围

$ 1 \le N \le 50000 $,  
$ 0 \le K \le 100000 $

输入样例：

```
100 7
1 101 1 
2 1 2
2 2 3 
2 3 3 
1 1 3 
2 3 1 
1 5 5
```

输出样例：

```
3
```

代码：

```cpp
#include <iostream>

using namespace std;

int n, k, D, x, y;
const int N = 5e4 + 10;
const int M = 3;
int p[N];
int d[N];

void init(){
    for(int i = 0; i <= n; i++){
        p[i] = i;
        d[i] = 0;
    }
}

int find(int x){
    if(x != p[x]){
        int t = find(p[x]);
        d[x] = (d[x] + d[p[x]]) % 3;
        p[x] = t;
    }
    return p[x];
}

bool D1(int x, int y){
    int p1 = find(x);
    int p2 = find(y);
    if(p1 == p2){
        return d[x] % M == d[y] % M;
    }
    p[p2] = p1;
    d[p2] = ((d[x] - d[y]) + M) % M;
    return true;
}

bool D2(int x, int y){
    int p1 = find(x);
    int p2 = find(y);
    if (p1 == p2) {
        return d[x] % M == (d[y]+1) % M;
    }
    p[p2] = p1;
    d[p2] = ((d[x]-d[y]-1) + M) % M;
    return true;
}

int main(){
    int res = 0;
    cin >> n;
    init();
    cin >> k;
    while(k --){
        cin >> D >> x >> y;
        if(x > n || y > n){
            res += 1;
        }
        else {
            if(D == 1){
                if(!D1(x, y)){
                    res += 1;
                }
            }
            if(D == 2){
                if(!D2(x, y)){
                    res += 1;
                }
            }
        }
    }
    cout << res;
    return 0;
}
```

### 哈希表

#### 模拟

##### 拉链法

```cpp
int h[N], e[N], ne[N], idx;

void insert(int x){
    int k = (x % N + N) % N;
    e[idx] = x;
    ne[idx] = h[k];
    h[k] = idx++;
}

bool find(int x){
    int k = (x % N + N) % N;
    for(int i = h[k]; i != -1; i = ne[i]){
        if(e[i] == x)
            return true;
    }
    return false;
}
```

##### 开放寻址法

```cpp
const int N = 200003, null = 0x3f3f3f3f;

int a[N];

int find(int x){
    int k = (x % N + N) % N;
    while(a[k] != null && a[k] != x){
        k ++;
        if(k == N) k = 0;
    }
    return k;
}
```

#### STL set/map

##### map

*//头文件* #include<map> *//初始化定义* map<string,string> mp; map<string,int> mp; map<int,node> mp;*//node是结构体类型*



代码	含义
mp.find(key)	返回键为key的映射的迭代器 O(logN) 注意：用find函数来定位数据出现位置，它返回一个迭代器。当数据存在时，返回数据所在位置的迭代器，数据不存在时，返回mp.end()
mp.erase(it)	删除迭代器对应的键和值O(1)
mp.erase(key)	根据映射的键删除键和值 O(logN)
mp.erase(first,last)	删除左闭右开区间迭代器对应的键和值 O(last-first)
mp.size()	返回映射的对数 O(1)
mp.clear()	清空map中的所有元素 O(N)
mp.insert()	插入元素，插入时要构造键值对
mp.empty()	如果map为空，返回true，否则返回false
mp.begin()	返回指向map第一个元素的迭代器（地址）
mp.end()	返回指向map尾部的迭代器（最后一个元素的下一个地址）
mp.rbegin()	返回指向map最后一个元素的反向迭代器（地址）
mp.rend()	返回指向map第一个元素前面(上一个）的反向迭代器（地址）
mp.count(key)	查看元素是否存在，因为map中键是唯一的，所以存在返回1，不存在返回0
mp.lower_bound()	返回一个迭代器，指向键值>= key的第一个元素
mp.upper_bound()	返回一个迭代器，指向键值> key的第一个元素

##### set

**2.set中常用的方法**

------

**begin()   　　 ,返回set容器的第一个元素**

**end() 　　　　 ,返回set容器的最后一个元素**

**clear()  　　   ,删除set容器中的所有的元素**

**empty() 　　　,判断set容器是否为空**

**max_size() 　 ,返回set容器可能包含的元素最大个数**

**size() 　　　　 ,返回当前set容器中的元素个数**

**rbegin　　　　 ,返回的值和end()相同**

**rend()　　　　 ,返回的值和rbegin()相同**

## 动态规划

### 最长上升子序列

1.朴素版本

```cpp
#include <iostream>

using namespace std;

const int N = 1010;
int n;
int a[N];
int dp[N]; //dp中存的是以a[i]结尾的子序列的集合中最长的子序列的长度

int main(){
    cin >> n;
    for(int i = 1; i <= n; i ++){
        scanf("%d", &a[i]);
    }
    for(int i = 1; i <= n; i ++){
        dp[i] = 1;
        for(int j = 1; j < i; j ++){
            if(a[j] < a[i]){
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }
    int ans = 0;
    for(int j = 1; j <= n; j ++) ans = max(ans, dp[j]);
    cout << ans;
    return 0;
}
```

2.1模拟栈版本

```cpp
#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;
int main(void) {
    int n; cin >> n;
    vector<int>arr(n);
    for (int i = 0; i < n; ++i)cin >> arr[i];

    vector<int>stk;//模拟堆栈
    stk.push_back(arr[0]);

    for (int i = 1; i < n; ++i) {
        if (arr[i] > stk.back())//如果该元素大于栈顶元素,将该元素入栈
            stk.push_back(arr[i]);
        else//替换掉第一个大于或者等于这个数字的那个数
            *lower_bound(stk.begin(), stk.end(), arr[i]) = arr[i];
    }
    cout << stk.size() << endl;
    return 0;
}
```

2.2二分写法

```cpp
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010;

int n;
int a[N];
int q[N];

int main()
{
    scanf("%d", &n);
    for (int i = 0; i < n; i ++ ) scanf("%d", &a[i]);

    int len = 0;
    for (int i = 0; i < n; i ++ )
    {
        int l = 0, r = len;
        while (l < r)
        {
            int mid = l + r + 1 >> 1;
            if (q[mid] < a[i]) l = mid;
            else r = mid - 1;
        }
        len = max(len, r + 1);
        q[r + 1] = a[i];
    }

    printf("%d\n", len);

    return 0;
}
```

