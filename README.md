# C++算法模板

## 基础算法

### 排序

#### 快速排序

```cpp
void quickSort(int q[], int l, int r) {
    if (l >= r) return;
    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while (i < j)
    {
        do i ++ ; while (q[i] < x);
        do j -- ; while (q[j] > x);
        if (i < j) swap(q[i], q[j]);
    }
    quickSort(q, l, j), quickSort(q, j + 1, r);
}
```

#### 归并排序

```cpp
void mergeSort(int q[], int l, int r) {
    if (l >= r) return;
    int mid = l + r >> 1;
    mergeSort(q, l, mid);
    mergeSort(q, mid + 1, r);
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
int bSearch_1(int l, int r) {
    while (l < r) {
        int mid = l + r >> 1;
        if (check(mid)) r = mid;    // check()判断mid是否满足性质
        else l = mid + 1;
    }
    return l;
}
// 区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
int bSearch_2(int l, int r) {
    while (l < r) {
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

double bSearch_3(double l, double r) {
    const double eps = 1e-6;   // eps 表示精度，取决于题目对精度的要求
    while (r - l > eps) {
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
vector<int> add(vector<int> &A, vector<int> &B) {
    if (A.size() < B.size()) return add(B, A);
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size(); i ++ ) {
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
vector<int> sub(vector<int> &A, vector<int> &B) {
    vector<int> C;
    for (int i = 0, t = 0; i < A.size(); i ++ ) {
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
vector<int> mul(vector<int> &A, int b) {
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ ) {
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
vector<int> div(vector<int> &A, int b, int &r) {
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- ) {
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
    for (int i = 0; i < lena; i++) {
        for (int j = 0; j < lenb; j++) {
            c[i + j] += a[i] * b[j] + jw;
            jw = c[i + j] / 10;
            c[i + j] %= 10;
        }
        c[i + lenb] = jw;
    }
    for (int i = lenc - 1; i >= 0; i--) {
        if (0 == c[i] && lenc > 1) lenc--;
        else break;
    }
    for (int i = lenc - 1; i >= 0; i--) cout << c[i];
    return 0;
}
```

### 离散化

```cpp
vector<int> alls; //存储所有待离散化的值
sort(alls.begin(), alls.end()); //将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end()); //去掉重复元素

//二分求出x对应的离散化的值
int find(int x) {
    int l = 0, r = alls.size() - 1;
    while(l < r) {
        int mid = l + r >> 1;
        if(alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1;
}
```

### 尺取法（双指针）

```cpp
for (int i = 0, j = 0; i < n; i ++ ) {
    while (j < i && check(i, j)) j ++ ;
    // 具体问题的逻辑
}
//常见问题分类：
    //(1) 对于一个序列，用两个指针维护一段区间
    //(2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

```

### RMQ(倍增)

```cpp
#include <iostream>
#include <cmath>

using namespace std;

const int N = 2e5 + 10, M = 20;

int arr[N];
int f[N][M];
int n;

void init(){
    for(int j = 0; j < M; j ++){
        for(int i = 1; i + (1 << j) - 1 <= n; i ++){
            if(!j) f[i][j] = arr[i];
            else f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
        }
    }
}

int query(int a, int b){
    int k = log(b - a + 1) / log(2);
    return max(f[a][k], f[b - (1 << k) + 1][k]);
}

int main(){
    cin >> n;
    for(int i = 1; i <= n; i ++) scanf("%d", &arr[i]);
    init();
    int m;
    cin >> m;
    while(m --){
        int a, b;
        scanf("%d%d", &a, &b);
        cout << query(a, b) << endl;
    }
    return 0;
}
```

## 数论与数学知识

### c++求上取整

$\lceil \frac{l}{p} \rceil = \lfloor \frac{l+p-1}{p} \rfloor$

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

#### 快速幂求逆元

```cpp
#include <iostream>

using namespace std;

typedef long long LL;

int n, a, p;

int q_pow(int a, int b, int k) {
    int ans = 1;
    while (b) {
        if (b & 1) ans = (LL)ans * a % k;
        a = (LL)a * a % k;
        b >>= 1;
    }
    return ans;
}

int main() {
    scanf("%d", &n);
    while (n --) {
        scanf("%d%d", &a, &p);
        int res = q_pow(a, p - 2, p);
        if (a % p) printf("%d\n", res);
        else puts("impossible");
    }
}

作者：syf666
链接：https://www.acwing.com/blog/content/10351/
来源：AcWing
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

#### 线性逆元

证明：

线性求解n个数字的逆元，需要找到新元素的逆元同以往求解过逆元的关系。
以下面式子举例,对于要求解逆元的k,模数为p，有：
		$ p=ak + b \\ (b < a,k)$

进而有：
		$ak + b \equiv 0\, (mod\,p)$

两边同时乘以k-1b-1,得到：

$ab^{-1} + k^{-1} \equiv 0\,(mod\,p) \\$

即
		$k^{-1}\equiv-ab^{-1}\,(mod\,p)$

我们知道，
		$a=\left \lfloor p \over k\right \rfloor \\$

$b=p\,mod\,k$

因此，有

$k^{-1}\equiv - \left \lfloor p\over k\right \rfloor(p\,mod\,k)^{-1}$

```cpp
inv[1] = 1;
for(int i = 2;i <= n;i++){
	inv[i] = (p - p / i) * inv[p % i] % p;
}
```

求解n个数字不同数字的逆元

求解n个不同数字的逆元，可以先维护一个前缀积，其最后一项是所有数字的乘积，求该项的逆元即求所有项逆元的乘积。由于逆元的特殊性质，逆元的乘积乘上其中某个元素即会消去对应的元素，因此我们可以借助前缀积来逐个迭代处理出所有数字的逆元。

$(∏\limits^n\limits_{i=1}a_i)^{−1}≡∏\limits^n\limits_{i=1}a_i^{-1}(mod\ p)$

或

$(a_1a_2...a_n)^{-1}\equiv a_1^{-1}a_2^{-1}...a_n^{-1}(mod\ p)$

且有

$∏\limits_{i=1}\limits^na^{−1}_i∗a_n≡∏\limits^{n-1}_{i=1}a_i^{-1}$

于是便可以处理处所有元素的逆元：

```cpp
/*s是前缀积，inv是逆元*/
s[1] = a[1];
/*计算前缀积*/
for(int i = 2;i <= n;i++){
	s[i] = s[i - 1] * a[i] % p;
}
/*处理所有元素乘积的逆元，使用快速幂发求解单个逆元*/
inv[n] = fpow(s[n],p - 2);
/*逆元的前缀积*/
for(int i = n - 1;i >= 1;i--){
	inv[i] = inv[i + 1] * a[i + 1] % p;
}
/*计算全部逆元*/
for(int i = 2;i <= n;i++){
	inv[i] = inv[i] * s[i - 1] % p;
}
————————————————
版权声明：本文为CSDN博主「卷儿~」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/wayne_lee_lwc/article/details/107870741
```

### 扩展欧几里得算法

```cpp
int exgcd(int a, int b, int &x, int &y)  // 扩展欧几里得算法, 求x, y，使得ax + by = gcd(a, b)
{
    if (!b)
    {
        x = 1; y = 0;
        return a;
    }
    //d始终都是a和b的最大公约数, 倒着传参是为了简化计算
    int d = exgcd(b, a % b, y, x);
    y -= (a / b) * x;
    return d;
    //返回d，使得能够在求最大公约数的同时完成x，y的凑整
}
```

### 高斯消元法

```cpp
int gauss() {
    int c, r;
    for (r = 1, c = 1; c <= n; c ++) {
        int t = r;
        for (int i = r + 1; i <= n; i ++)
            if (fabs(a[i][c]) - fabs(a[t][c]) > eps) t = i;
        if (fabs(a[t][c]) < eps) continue;
        if (t != r) swap(a[t], a[r]);
        for (int i = n + 1; i >= c; i --) a[r][i] /= a[r][c];
        for (int i = r + 1; i <= n; i ++) {
            if (fabs(a[i][c]) > eps)
                for (int j = n + 1; j >= c; j --)
                    a[i][j] -= a[r][j] * a[i][c];
        }
        r ++;
    }
    if (r <= n) {
        for (int i = r; i <= n; i ++) {
            if (fabs(a[i][n + 1]) > eps) return 2; // 代表有无穷多组解
        }
        return 1; // 代表无解
    }
    for (int i = n; i >= 1; i --) {
        for (int j = i + 1; j <= n; j ++) {
            a[i][n + 1] -= a[i][j] * a[j][n + 1];
        }
    } // 代表有唯一组解
    return 0;
}
```

### 筛质数

#### 朴素筛法

```cpp
void getPrimes(int n) {
    for (int i = 2; i <= n; i ++) {
        if (!st[i]) {
            primes[cnt ++] = n;
        }
        for (int j = i + i; i <= n; j += i) st[j] = true;
    }
}
```

时间复杂度分析：

每个数的倍数都被筛掉了，因此时间复杂度为：

$\frac{n}{2}+\frac{n}{3}+\dots+\frac{n}{n}$

$=n(\frac{1}{2}+\frac{1}{3}+...+\frac{1}{n})$

$=n(ln(n)+c)$	其中c是一个欧拉常数，是一个无限不循环小数，值是0.577左右

#### 质数定理

1~n中有$\frac{n}{ln(n)}$个质数

我们可以只用把质数的倍数删掉，这样粗劣的时间复杂度就是$\frac{n(ln(n))}{ln(n)}$，但是这只是个粗略的估计，这个估计是不对的，真实的时间复杂度是$O(nlog(log(n)))$。

这个算法也就是下面的埃氏筛法。

#### 埃氏筛法

```cpp
//O(nloglogn)
int getPrimes(int n){
    int idx = 1;
    for(int i = 2; i <= n; i ++){
        if(!st[i]){
            prime[idx++] = i;
            for(int j = i + i; j <= n; j += i) st[j] = 1;
        }
    }
    return idx - 1;
}
```

#### 线性筛法

线性筛法在10^7^次方的情况下会比埃氏筛法快一倍左右，在10^6^的情况下两个算法速度差不多。

整体思路：n只会被它的最小质因子筛掉

```cpp
void getPrimes(int n){
    for(int i = 2; i <= n; i++){
        if(!st[i]) primes[cnt++] = i; 
        // primes[j] <= n / i即primes * i <= n，即x <= n
        for(int j = 0; primes[j] <= n / i; j++){
            // 当还没有执行到下面的break语句时
            // 由于我们是从小到大枚举所有质数的，且还没有枚举到i的最小质因子
            // 所以当前的primes[j]一定是primes[j] * i的最小质因子
            // 所以就把primes[j] * i筛掉
            st[primes[j] * i] = true;
            // 当这段代码执行的时候就意味着primes[j]一定是i的最小质因子
            // primes[j]是从小到大枚举的质数，当第一次满足i % primes[j] == 0的时候
            // 说明primes[j]就一定是i的最小质因子
            // 同时此时primes[j]也一定是primes[j] * i的最小质因子
            // 因此上面的筛成立
            // 当如果此时再继续向后枚举质数，接下来的质数就不是primes[j] * i的最小质因子了
            // 因此要即时break掉
            if(i % primes[j] == 0) break;
        }
    }
}
```

证明线性：

对于一个合数x，假设primes[j]是x的最小质因子，当i枚举到x/primes[j]的时候，我们就可以在4到18行把x给筛掉。

第五行的判断条件不需要加j <= cnt

因为当i是合数的时候，primes[j]一定会在枚举到i的最小质因子时停下来，而i的最小质因子一定是小于i的，会在i之前被标记为primes，放到primes数组中来。

当i是质数的时候，当primes[j]=i的时候，枚举也会停下来。（是在break的时候停下来）

### 欧拉函数

$\phi(n)$: 1~n中和n互质的数的个数

$\phi(6)=2$

一个数可以写成很多个质数的乘积的形式：

$N=P_1^{\alpha_1}P_2^{\alpha_2}...P_k^{\alpha_k}$ 

$\phi(N)=N(1-\frac{1}{p_1})(1-\frac{1}{p_2})...(1-\frac{1}{{p_k}})$

上式的证明会用到容斥原理。

那么如何计算从1~N中和N互质的数的个数呢

- 从1~N中去掉p1，p2，…，pk的所有倍数

  这里面就会多去一部分数，比如一个数可能既是p1的倍数又是p2的倍数

- 加上所有pi*pj的倍数

  $N-\frac N {P_1} - \frac {N}{P_1} - ... - \frac N {P_1} + \frac N {P_1P_2} + \frac N {P_1 P_3} + ...$

- 减去所有三个质数的倍数

  …

然后把最上面的式子展开，会发现这两个式子是相等的。

时间复杂度：时间复杂度瓶颈在分解质因数上，分解质因数的时间复杂度是$O(\sqrt N)$

代码：

```cpp
int euler(int a) {
    int res = a;
    for (int i = 2; i <= a / i; i ++) {
        if (a % i == 0) {
            res = res / i * (i - 1);
            while (a % i == 0) a /= i;
        }
    }
    if (a > 1) res = res / a * (a - 1);
    return res;
}
```

#### 筛法求欧拉函数

$φ(ab)= \frac {φ(a)φ(b)gcd(a,b)} {φ(gcd(a,b))}$

```cpp
int getEuler(int n){
    phi[1] = 1;
    for(int i = 2; i <= n; i ++){
        if(!st[i]){
            prime[cnt ++] = i;
            phi[i] = i - 1;
        }
        for(int j = 0; prime[j] <= n / i; j ++){
            st[prime[j] * i] = true;
            if(i % prime[j] == 0){
                // 此时pj是i的最小质因子，i的质因子中有j了，所以pj*i的所有(1-pj*i的质因子)都在phi[i]中计算过了，因此两者的区别就只有式子开头的N，所以phi[pj * i] = phi[i] * prime[j]
                phi[prime[j] * i] = phi[i] * prime[j];
                break;
            }
            // 这里pj - 1就是(1 - 1 / pj) * pj
            phi[prime[j] * i] = phi[i] * (prime[j] - 1);
        }
    }
    int res = 0;
    for(int i = 1; i <= n; i ++){
        res += phi[i];
    }
    return res;
}
```

### 组合数

#### 组合数1

主要思想：$C_a^b = C_{a-1}^{b-1} \times C_{a - 1}^{b}$

时间复杂度：O(n^2)

```cpp
#include <iostream>

using namespace std;

const int N = 2010, mod = 1e9 + 7;
int c[N][N];

void init(){
    for(int i = 0; i < N; i ++){
        for(int j = 0; j <= i; j ++){
            if(j == 0) c[i][j] = 1;
            else c[i][j] = (c[i - 1][j] + c[i - 1][j - 1]) % mod;
        }
    }
}

int main(){
    int n;
    cin >> n;
    init();
    while(n --){
        int a, b;
        scanf("%d%d", &a, &b);
        printf("%d\n", c[a][b]);
    }
    return 0;
}
```

#### 组合数2

预处理阶乘

```cpp
#include <iostream>

using namespace std;

typedef long long LL;

const int N = 100010, mod = 1e9 + 7;
int fact[N], infact[N];//存的分别是i的阶乘及其逆元

int qmi(int a, int k, int p)  // 求a^k mod p
{
    int res = 1;
    while (k)
    {
        if (k & 1) res = (LL)res * a % p;
        a = (LL)a * a % p;
        k >>= 1;
    }
    return res;
}

int main(){
    fact[0] = infact[0] = 1;
    for(int i = 1; i < N; i ++){
        fact[i] = (LL) fact[i - 1] * i % mod;
        infact[i] = (LL) infact[i - 1] * qmi(i, mod - 2, mod) % mod;
    }
    int n;
    cin >> n;
    while(n --){
        int a, b;
        scanf("%d%d", &a, &b);
        //这里要及时取模，否则会爆long long
        printf("%d\n", (LL) fact[a] * infact[b] % mod * infact[a - b] % mod);
    }
    return 0;
}
```

### 博弈论

必胜态：从这个状态总是有某种方式走到一个必败态

必败态：从这个状态无论怎么走都是会走到必胜态

如果先手初始处于必败态，那么先手必败，反之先手必胜

#### SG函数

首先，我们定义一个mex函数（或者称为mex操作）：

$mex(A)$: 返回A集合中未曾出现过的最小的自然数

$sg(x)$: 如果处于最终态，则返回0；否则返回该节点可以到达的状态（A集合）中未曾出现过的最小的自然数

## 图论

### 最短路

#### Dijkstra求最短路1

Dijkstra算法总体流程：

1. dist[1] = 0, dist[i] = +$\infty$
2. for i : 1 ~ n
   - t$\leftarrow$不在s中的距离最近的点（其中s表示当前已确定最短距离的点的集合）$\color{green}{O(n^2)}$
   - s$\leftarrow$ t $\ \color{green}{n \times O(1)}$
   - 用t更新其他点的距离 $\ \color{green}{O(m)}$

朴素版dijkstra算法时间复杂度是O（$n^2$）（其中n代表图中点的个数）

<font color=red>因此适合用于稠密图，即边多点少的图</font>

```cpp
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace std;

const int N = 510;
int n, m;
int g[N][N]; //存储图，g[i][j]的值是i到j的距离，因为是稠密图，所以用邻接矩阵来存
int dist[N]; //dist[i]存的是i节点到1节点的最短距离
int st[N];   //st[i]为true说明i节点的值是最小距离

int dijkstra(){
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    //循环n-1次，即更新n-1次st[?]，这样就把1到1~n这n个节点的最短路都算出来了
    for(int i = 1; i <= n - 1; i ++){
        int t = -1;
        //找到未存入最短值且到1节点距离最短的节点t
        for(int j = 1; j <= n; j++){
            if(!st[j] && (t == -1 || dist[t] > dist[j]))
                t = j;
        }
        //if(t == n) goto end;这个优化只有这题能加，意思是当求得1到n的最短路时，即可break，因为这就是题目所求
        st[t] = true;
        //用t节点来更新其他节点（不是t的邻点的话他的g[t][j]就为无穷）
        for(int j = 1; j <= n; j ++){
            dist[j] = min(dist[j], dist[t] + g[t][j]);
        }
    }
    //end:
    if(dist[n] == 0x3f3f3f3f) return -1;
    return dist[n];
}

int main(){
    scanf("%d%d", &n, &m);
    memset(g, 0x3f, sizeof g);
    while(m --){
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        g[a][b] = min(g[a][b], c); //这里是用来处理重边的情况，重边取最短边即可
    }
    printf("%d\n", dijkstra());
    return 0;
}
```

#### 堆优化的Dijkstra算法

堆优化版Dijkstra算法总体流程：

1. dist[1] = 0, dist[i] = +$\infty$
2. for i : 1 ~ n
   - t$\leftarrow$不在s中的距离最近的点（其中s表示当前已确定最短距离的点的集合）$\color{green}{O(n)}$
   - s$\leftarrow$ t $\ \color{green}{O(n)}$
   - 用t更新其他点的距离 $\ \color{green}{O(mlog(n))}$

堆优化版dijkstra算法时间复杂度是O（$mlog(n)$）（其中n代表图中点的个数, m代表边的个数）

$\color {red} {因此适合用于稀疏图，即边少点多的图}$

```cpp
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <queue>
#include <cstring>

using namespace std;

typedef pair<int, int> PII;

const int N = 150010;
int h[N], e[N], ne[N], idx;
int w[N];
int dist[N];
bool st[N];
int n, m;

void add(int x, int y, int c){
    w[idx] = c;
    e[idx] = y;
    ne[idx] = h[x];
    h[x] = idx++;
}

int dijkstra(){
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;
    priority_queue<PII, vector<PII>, greater<PII>> heap;
    heap.push({0, 1});
    while(heap.size()){
        PII k = heap.top();
        heap.pop();
        int ver = k.second, distance = k.first;
        if(st[ver]) continue;
        st[ver] = true;
        for(int i = h[ver]; i != -1; i = ne[i]){
            int j = e[i];
            if(dist[j] > distance + w[i]){
                dist[j] = distance + w[i];
                heap.push({dist[j], j});
            }
        }
    }
    if(dist[n] == 0x3f3f3f3f) return -1;
    else return dist[n];
}

int main(){
    memset(h, -1, sizeof h);
    scanf("%d%d", &n, &m);
    while(m --){
        int x, y, c;
        scanf("%d%d%d", &x, &y, &c);
        add(x, y, c);
    }
    printf("%d\n", dijkstra());
    return 0;
}
```

#### SPFA

spfa算法

底子是： for 所有边a，b，w  
   $\ \ \ \ \ \ \ \ \ \ \ \ \ \ $dist[b] = min(dist[b], dist[a] + w)
流程是：
    queue$\leftarrow$1
    while queue不空：
    1. t$\leftarrow$queue.front
        $\ \ \ \ $q.pop
        2. 更新t的所有出边t$\stackrel{w}{\longrightarrow}$b
        $\ \ \ \ $q$\leftarrow$b

$\color{red}{spfa求最短路不能处理有自环负权和负权环的情况，这两种情况会在while循环中卡住，所以这题数据不太强}$

1.stl queue

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>
#include <queue>

using namespace std;

const int N = 100010;

int n, m;
int h[N], w[N], e[N], ne[N], idx;
int dist[N];
bool st[N]; // st数组是判断当前节点是否在队列当中，防止重复入队

void add(int a, int b, int c)
{
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx ++ ;
}

int spfa()
{
    memset(dist, 0x3f, sizeof dist);
    dist[1] = 0;

    queue<int> q;
    q.push(1);
    st[1] = true;

    while (q.size())
    {
        int t = q.front();
        q.pop();

        st[t] = false;

        for (int i = h[t]; i != -1; i = ne[i])
        {
            int j = e[i];
            if (dist[j] > dist[t] + w[i])
            {
                dist[j] = dist[t] + w[i];
                if (!st[j])
                {
                    q.push(j);
                    st[j] = true;
                }
            }
        }
    }

    return dist[n];
}

int main()
{
    scanf("%d%d", &n, &m);

    memset(h, -1, sizeof h);

    while (m -- )
    {
        int a, b, c;
        scanf("%d%d%d", &a, &b, &c);
        add(a, b, c);
    }

    int t = spfa();

    if (t == 0x3f3f3f3f) puts("impossible");
    else printf("%d\n", t);

    return 0;
}

```

循环队列

```cpp
#include <iostream>
#include <cstring>

using namespace std;

const int N = 2510, M = 6200 * 2 + 10;

int n, m, S, T;
int h[N], e[M], w[M], ne[M], idx;
int dist[N], q[N];
bool st[N];

void add(int a, int b, int c){
    e[idx] = b, w[idx] = c, ne[idx] = h[a], h[a] = idx++;
}

void spfa(){
    memset(dist, 0x3f, sizeof dist);
    dist[S] = 0;
    int hh = 0, tt = 1;
    q[0] = S, st[S] = true;
    while(hh != tt){
        int t = q[hh++];
        if(hh == N) hh = 0;
        st[t] = false;
        for(int i = h[t]; ~i; i = ne[i]){
            int j = e[i];
            if(dist[j] > dist[t] + w[i]){
                dist[j] = dist[t] + w[i];
                if(!st[j]){
                    q[tt++] = j;
                    if(tt == N) tt = 0;
                    st[j] = true;
                }
            }
        }
    }
}

int main(){
    cin >> n >> m >> S >> T;
    memset(h, -1, sizeof h);
    for(int i = 0; i < m; i ++){
        int a, b, c;
        cin >> a >> b >> c;
        add(a, b, c), add(b, a, c);
    }
    spfa();
    cout << dist[T] << endl;
    return 0;
}
```

### 最小生成树

#### kruskal算法

```cpp
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace std;

const int N = 100010, M = 200010, INF = 0x3f3f3f3f;

int n, m;
int p[N];

struct Edge {
    int a, b, w;
    bool operator<(const Edge &W) const {
        return w < W.w;
    }
}edges[M];

int find(int x) {
    if (p[x] != x) p[x] = find(p[x]);    // 路径压缩
    return p[x];
}

int kruskal() {
    sort(edges, edges + m);    // 将边按照权重从小到大排序
    for (int i = 1; i <= n; i ++ ) p[i] = i;    // 初始化并查集
    int res = 0, cnt = 0;
    for (int i = 0; i < m; i ++ )
    {
        int a = edges[i].a, b = edges[i].b, w = edges[i].w;

        a = find(a), b = find(b);
        if (a != b)    // 判断两个节点是否在同一个连通块中
        {
            p[a] = b;    // 将两个连通块合并
            res += w;    // 将边权重累加到结果中
            cnt ++ ;     // 记录加入生成树的边的数量
        }
    }
    if (cnt < n - 1) return INF;    // 判断生成树中是否包含n-1条边
    return res;
}

int main() {
    scanf("%d%d", &n, &m);

    for (int i = 0; i < m; i ++ )
    {
        int a, b, w;
        scanf("%d%d%d", &a, &b, &w);
        edges[i] = {a, b, w};
    }

    int t = kruskal();

    if (t == INF) puts("impossible");    // 无法生成最小生成树
    else printf("%d\n", t);             // 输出最小生成树的权重

    return 0;
}
```

Kruskal算法是一种贪心算法，其正确性可以通过贪心选择性质的证明得到。

假设我们要构造一个无向连通图的最小生成树。我们定义一个边集$E$为“好的”，如果它是某棵最小生成树的子集。我们定义一个边集$E$为“坏的”，如果它包含了某个环上的边。

首先我们考虑一个引理：对于任何的连通图，$E$为“好的”，则$E$中的边是没有环的。

证明：如果$E$中存在环，则我们可以从中删去任意一条边，得到一个更小的$E$集合，仍然满足“好的”性质。因此，在“好的”边集中，所有的边都没有构成环。

然后我们考虑另一个引理：对于任何的连通图，令$E$为“好的”边集合，令$e$为一条“好的”边，并令$E'=E \backslash {e}$。则$E'$为某棵生成树的子集。

证明：因为$e$为“好的”边，所以$E$中不包含$e$的时候，$E$仍然是连通的。因此，$E'$也是连通的。由于$E'$中没有环，所以它也是无向无环图，也就是一棵树。此外，$E'$中有$n-1$条边，与任何生成树的边数相同，所以$E'$也是某棵生成树的子集。

有了上面的两个引理，我们可以得到Kruskal算法的正确性证明：

1. 初始化$E$为空集；
2. 将所有边按照权值从小到大排序；
3. 依次考虑每条边$e$，如果将其加入$E$不会产生环，则将其加入$E$中；
4. 当加入了$n-1$条边后，停止算法。此时$E$即为一棵生成树。

根据引理1，$E$中所有的边没有环；根据引理2，算法停止时，$E$为某棵生成树的子集；根据Kruskal算法的贪心选择性质，我们知道$E$为全局最优的“好的”边集合，因此$E$中的边必定构成了一棵最小生成树。

因此，Kruskal算法的正确性得证。

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

void push(int x) {
    q[++tt] = x;
}

void pop() {
    hh++;
}

int front() {
    return q[hh];
}
```

#### 循环队列

主要在bfs及spfa算法中使用

```cpp
void bfs(int x) {
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

附：最长不下降子序列

```cpp
void subseq1(int a[], int l, int r){ // 对原数组的每个下标i(从1开始)，求以a[i]结尾的最长不下降子序列长度
    int len = 0;
    // q下标从2开始才放如元素,q[i]是有长度为i-1的子序列的最小的末尾值
    for(int i = l; i <= r; i ++){
        int ll = l, rr = len + 1;
        while(ll < rr){
            int mid = ll + rr + 1 >> 1;
            if(q[mid] <= a[i]) ll = mid;
            else rr = mid - 1;
        }
        len = max(len, rr);
        q[rr + 1] = a[i];
    }
    // 二分求出以a[i]结尾的最长不下降子序列长度
    for(int i = l; i <= r; i ++){
        int l2 = 2, r2 = len + 1;
        while(l2 < r2){
            int mid = l2 + r2 + 1 >> 1;
            if(q[mid] <= a[i]) l2 = mid;
            else r2 = mid - 1;
        }
        dp1[i] = l2 - 1;
    }
}
```

### 背包问题

#### 01背包

状态表示：$f[i][j]$:在前i个物品中选，总体积不超过j的最大价值。

##### 01背包问题的二维代码

```cpp
{
    //动态规划问题一般下标从1开始，便于从0的情况递推结果。
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= m; j ++) {
            f[i][j] = f[i - 1][j];
            if (j >= v[i]) {
                f[i][j] = max(f[i][j], f[i - 1][j - v[i]] + w[i]);
            }
        }
    }
}
```

##### 01背包一维优化

```cpp
{
    for (int i = 1; i <= n; i ++) {
        for (int j = m; j >= v[i]; j --) {
            f[j] = max(f[j], f[j - v[i]] + w[i]);
        }
    }
}
```

#### 完全背包问题

特征：一个物品可以选无数次。

状态表示：$f[i][j]$表示从前i个物品里选择（一个物品可以选多次），总体积不超过j的最大价值。

##### 完全背包问题的三重循环代码

```cpp
{
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= m; j ++) {
            for (int k = 0; k * v[i] <= j; k ++) {
                f[i][j] = max(f[i][j], f[i - 1][j - k * v[i]] + k * w[i]);
            }
        }
    }
}
```

##### 二重循环二维的优化过程

```cpp
// 01
f[i][j]        = max(f[i][j],            f[i - 1][j - k * v[i]] + k * w[i]);
// 完全
f[i][j]        = max(f[i - 1][j],        f[i - 1][j - v[i]] + w[i], f[i - 1][j - 2 * v[i]] + 2 * w[i], .....)
f[i][j - v[i]] = max(f[i - 1][j - v[i]], f[i - 1][j - 2 * v[i]] + 1 * w[i], .....)
```
三式代入二式，得：
<font color=red>`f[i][j] = max(f[i - 1][j],f[i][j - v[i]] + w[i]);`</font>

```cpp
{
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= m; j ++) {
            f[i][j] = f[i - 1][j];
            if (j >= v[i]) {
                f[i][j] = max(f[i][j], f[i][j - v[i]] + w[i]);
            }
        }
    }
}
```

##### 完全背包一维最终优化

```cpp
{
    for (int i = 1; i <= n; i ++) {
        for (int j = v[i]; j <= m; j ++) {
            f[j] = max(f[j], f[j - v[i]] + w[i]);
        }
    }
}
```

#### 多重背包1

特征：一个物品有严格的次数限制。
状态表示：$f[i][j]$表示从前i个物品里选择（一个物品有严格的次数限制），总体积不超过j的价值；

##### 多重背包问题的三重循环代码

```cpp
//s[i]表示第i种物品共有s[i]个
int f[N][N], s[N], v[N], w[N];

{
    for (int i = 1; i <= n; i ++) {
        for (int j = 1; j <= m; j ++) {
            for (int k = 0; k <= s[i] && k * v[i] <= j; k ++) {
                f[i][j] = max(f[i][j], f[i - 1][j - k * v[i]] + k * w[i]);
            }
        }
    }
}
```

##### 多重背包问题一维优化代码

```cpp
{
    for (int i = 1; i <= n; i ++) {
        for (int j = m; j >= v[i]; j --) {
            for (int k = 0; k <= s[i] && k * v[i] <= j; k ++) {
                f[j] = max(f[j], f[j - k * v[i]]+ k * w[i]);
            }
        }
    }
}
```

##### 多重背包问题打包版

利用二进制优化的思想，把同一种类型的物品按照2的不同次幂个为一组进行打包，从而优化时间复杂度。

时间复杂度：$O(nmlogs)$

```cpp
// s[i]存储第i种物品的个数
// a, b数组存的是原体积和原价值
int a[N], b[N];
int v[N], w[N], s[N];

{
    int cnt = 0;
    for (int i = 1; i <= n; i ++) {
        int k = 1;
        int ss = s[i];
        while (k <= s) {
            cnt ++;
            v[cnt] = k * a[i];
            w[cnt] = k * b[i];
            s -= k;
            k *= 2;
        }
        if (s > 0) {
            cnt ++;
            v[cnt] = s * a;
            w[cnt] = s * b;
        }
    }
    for (int i = 1; i <= cnt; i ++) {
        for (int j = m; j >= v[i]; j --) {
            f[j] = max(f[j], f[j - v[i]] + w[i]);
        }
    }
}
```

##### 多重背包单调队列优化

时间复杂度：$O(nm)$

<img src="https://cdn.acwing.com/media/article/image/2023/04/15/226750_2e51f2afdb-_~948ZKD$1S{~4(4~{LS_Z2.png">

```cpp
int f[N];
int g[N]; // g是f的滚动数组
int v[N], w[N], s[N];

{
    for (int i = 1; i <= n; i ++) {
        memcpy(g, f, sizeof f); //把上一层的f存在g里
        for (int j = 0; j < v[i]; j ++) {
            int hh = 0, tt = -1;
            //滑动窗口长度为s，不包括f[k]本身，因此f[k] = max(f[k], max(others))
            for (int k = j; k <= m; k += v[i]) {
                if (hh <= tt && q[hh] < k - s[i] * v[i]) hh ++;
                if (hh <= tt) f[k] = max(f[k], g[q[hh]] + (k - q[hh]) / v[i] * w[i]);
                //由图中表达式知g数组中的值都是不包含w的，这里我们进行比较只需比较绝对值
                //因此只要营造出递减数列的效果即可
                while (hh <= tt && g[q[tt]] - (q[tt] - j) / v[i] * w[i] <= g[k] - (k - j) / v[i] * w[i]) tt --;
                q[++tt] = k;
            }
        }
    }
}

```

#### 分组背包问题

特征：一组物品只能选择其中的一个。

状态表示：f\[i][j]表示从前i组物品里选择(一组物品只能选择其中的一个)，总体积不超过j的最大价值；
状态计算：<font color=red>`f[i][j]=max(f[i-1][j],f[i-1][j-w[i][k]]+v[i][k])`</font>(解释；表示从前i组物品里选择体积不超过j的物品价值的最大值是从要么不选这个物品，和要么选择这个组中第k个物品的价值之间进行选择)

##### 分组背包问题一维优化代码

```cpp
// w[i][j]代表第i组物品中的第j个的价值，v同
int w[N][N],v[N][N];
// s[i]代表第i组物品有多少个
int f[N],s[N];

{
    for (int i = 1; i <= n; i ++) {
        for (int j = m; j >= 0; j --) {
            for (int k = 1; k <= s[i]; k ++) {
                if (v[i][k] <= j) {
                    f[j] = max(f[j], f[j - v[i][k]] + w[i][k]);
                }
            }
        }
    }
    return 0;
}
```

#### 背包问题中总体积最多/恰好/最少为j, 以求最大值为例(3是最小值)

1. 体积最多为j：`memset(f, 0, sizeof f), v >= 0`

2. 体积恰好为j：`memset(f, -0x3f, sizeof f), ,f[0] = 0, v >= 0`

- <font color=red> 怎么保证是恰好装满，通过初始化来实现, 如果不存在使f[i]满足是恰好装满的，那么这个f[i] = -INF, 即一个非法答案，就不会影响最终的计算。
</font>

3. 体积至少为j：`memset(f, -0x3f, sizeof f), f[0] = 0`

- f[i][j]: 从前i件物品中选，总体积至少为j的最小价值。

```cpp
memset(f, 0x3f, sizeof f);

{
    for (int i = 1; i <= n; i ++) {
        for (int j = m; j >= 0; j --) {
            f[j] = min(f[j], f[max(0, j - v[i])] + w[i]);
        }
    }
}
```

#### 有依赖的背包问题

$f[u][j]$: 表示在以u为根的子树中选，且先选出u（因为其他节点都依赖u, 这样表示状态会简单一点，可以往分组背包上面套），总体积不超过j的最大价值。

```cpp
void dfs(int u) {
    // 相当于分组背包问题中的物品组
    for (int u = h[u]; ~i; i = ne[i]) {
        int son = e[i];
        dfs(son);
        // 先求一下所有子树中的最大价值, 故先在最大体积中把当前节点去掉
        // 枚举体积
        for (int j = m - v[u]; j >= 0; j --) {
            // 枚举决策，这里以选取体积为多少讨论决策
            for (int k = 0; k <= j; k ++) {
                f[u][j] = max(f[u][j], f[u][j - k] + f[son][k]);
            }
        }
    }
    // 最后记得把当前节点加上
    for (int j = m; j >= v[u]; j --) f[u][j] = f[u][j - v[u]] + w[u];
    for (int j = v[u] - 1; j >= 0; j --) f[u][j] = 0;
}

dfs(root);
```

# 琐碎知识

## 等差数列各项平方求和公式

设首项为a1,公差为d的等差数列各项平方的和为：

=a1²+(a1+d)²+(a1+2d)²+--------+[a1+(n-1)d]²

=na1²+[2+4+6+-------+2(n-1)]d+[1²+2²+3²+-----+(n-1)²]d²

=na1²+n(n-1)d+n(n-1)(2n-1)d²

## 系数乘键值积排序求和定理（我瞎起的）

数组a[i]和数组b[i]

前者正向排序，后者逆向排序

然后每个元素乘积相加，这样得到的结果最小（前提a和b是非负数组）

