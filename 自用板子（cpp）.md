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

## 快读写模板

### 快读

```cpp
inline int read()
{
	int x = 0,f = 1;
	char ch = getchar();
	while (ch < '0' || ch>'9')
	{
		if (ch == '-')
			f = -1;
		ch = getchar();
	}
	while (ch >= '0' && ch <= '9')
	{
		x = (x << 1) + (x << 3) + (ch ^ 48);
		ch = getchar();
	}
	return x * f;
}
————————————————
版权声明：本文为CSDN博主「Utozyz」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_60404548/article/details/125676237
```

### 快写

```cpp
inline void write(int x)
{
	if (x < 0) putchar('-'), x = -x;
	if(x > 9)
		write(x / 10);
	putchar(x % 10 + '0');
	return;
}
```

### 优化

```cpp
#include <iostream>
using namespace std;
typedef long long LL;

inline LL read()
{
	LL x = 0, f = 1;
	char ch = getchar();
	while (!isdigit(ch))
	{
		if (ch == '-') 
			f = -1;
		ch = getchar();
	}
	while (isdigit(ch))
	{
		x = (x << 1) + (x << 3) + (ch ^ 48);
		ch = getchar();
	}
	return x * f;
}

inline void write(LL x)
{
	if (x < 0) putchar('-'), x = -x;
	if (x > 9) write(x / 10);
	putchar(x % 10 + '0');
}

int main()
{
	int a = read();
	write(a);
	return 0;
}

------------------------------------------------

版权声明：本文为CSDN博主「Utozyz」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_60404548/article/details/125676237
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

#### 冒泡排序

```cpp
int a[N];
for(int i = 1; i <= n - 1; i ++){
    for(int j = n - 1; j >= 1; j --){
        if(a[j] > a[j - 1]) swap(a[j], a[j - 1]);
    }
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

```cpp
vector<int> alls; //存储所有待离散化的值
sort(alls.begin(), alls.end()); //将所有值排序
alls.erase(unique(alls.begin(), alls.end()), alls.end()) //去掉重复元素

//二分求出x对应的离散化的值
int find(int x){
    int l = 0, r = alls.size() - 1;
    while(l < r){
        int mid = l + r >> 1;
        if(alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return r + 1;
}
```

### 尺取法（双指针）

```cpp
for (int i = 0, j = 0; i < n; i ++ )
{
    while (j < i && check(i, j)) j ++ ;

    // 具体问题的逻辑
}
//常见问题分类：
    //(1) 对于一个序列，用两个指针维护一段区间
    //(2) 对于两个序列，维护某种次序，比如归并排序中合并两个有序序列的操作

```

### RMQ(倍增)

https://www.acwing.com/problem/content/1275/

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

### 矩阵

#### 矩阵快速幂

```cpp
struct matrix{ int m[N][N]; };     //定义矩阵，常数N是矩阵的行数和列数
matrix operator * (const matrix& a, const matrix& b){   //重载*为矩阵乘法。注意const
    matrix c;   
    memset(c.m, 0, sizeof(c.m));  //清零
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            for(int k = 0; k<N; k++)
              //c.m[i][j] += a.m[i][k] * b.m[k][j];                   //不取模
                c.m[i][j] = (c.m[i][j] + a.m[i][k] * b.m[k][j]) % mod;//取模
    return c;
}
matrix pow_matrix(matrix a, int n){  //矩阵快速幂，代码和普通快速幂几乎一样
    matrix ans;   
    memset(ans.m,0,sizeof(ans.m));
    for(int i=0;i<N;i++)  ans.m[i][i] = 1; //初始化为单位矩阵，类似普通快速幂的ans=1
    while(n) {
        if(n&1) ans = ans * a;       //不能简写为ans *= a，这里的*重载了
        a = a * a;
        n>>=1;
    }
    return ans;
}

```

#### 矩阵乘法与路径问题

##### 例1. Cow Relays poj 3613

```cpp
#include<cstdio>
#include<algorithm>
#include<cstring>
const int INF=0x3f;
const int N=120;
int Hash[1005],cnt=0;                //用于离散化
struct matrix{int m[N][N]; };        //定义矩阵
matrix operator *(const matrix& a, const matrix& b){   //定义广义矩阵乘法
    matrix c;
    memset(c.m,INF,sizeof c.m);
    for(int i=1;i<=cnt;i++)          //i、j、k可以颠倒，因为对c来说都一样
        for(int j=1;j<=cnt;j++)
            for(int k=1;k<=cnt;k++)
               c.m[i][j] = std::min(c.m[i][j], a.m[i][k] + b.m[k][j]);
    return c;
}
matrix pow_matrix(matrix a, int n){  //矩阵快速幂，几乎就是标准的快速幂写法
    matrix ans = a;                  //矩阵初值ans = M^1
    n--;                             //上一行ans= M^1多了一次
    while(n) {                       //矩阵乘法：M^n
        if(n&1) ans = ans * a;
        a = a * a;
        n>>=1;
    }
    return ans;
}
int main(){
    int n,t,s,e;    scanf("%d%d%d%d",&n,&t,&s,&e);
    matrix a;                                //用矩阵存图
    memset(a.m,INF,sizeof a.m);
    while(t--){
        int u,v,w;     scanf("%d%d%d",&w,&u,&v);
        if(!Hash[u])   Hash[u] = ++cnt;      //对点离散化.  cnt就是新的点编号
        if(!Hash[v])   Hash[v] = ++cnt;
        a.m[Hash[u]][Hash[v]] = a.m[Hash[v]][Hash[u]] = w;
    }
    matrix ans = pow_matrix(a,n);
    printf("%d",ans.m[Hash[s]][Hash[e]]);
    return 0;
}

```

### 高斯-约当消元法

#### 例1. 高斯消元法 洛谷P3389

```cpp
//改写自：https://www.luogu.com.cn/blog/tbr-blog/solution-p3389
#include<bits/stdc++.h>
using namespace std;
double a[105][105];
double eps = 1e-7;
int main(){
	int n; scanf("%d",&n);
	for(int i=1;i<=n;++i)
		for(int j=1;j<=n+1;++j) 	scanf("%lf",&a[i][j]);
	for(int i=1;i<=n;++i){            //枚举列
		int max=i;
		for(int j=i+1;j<=n;++j)      //选出该列最大系数，真实目的是选一个非0系数
			if(fabs(a[j][i])>fabs(a[max][i]))   max=j;
		for(int j=1;j<=n+1;++j) swap(a[i][j],a[max][j]); //移到前面
		if(fabs(a[i][i]) < eps){     //对角线上的主元系数等于0,说明没有唯一解
			puts("No Solution");
			return 0;
		}
		for(int j=n+1;j>=1;j--)	  a[i][j]= a[i][j]/a[i][i]; //把这一行的主元系数变为1
		for(int j=1;j<=n;++j){       //消去主元所在列的其他行的主元
			if(j!=i)	{
				double temp=a[j][i]/a[i][i];
				for(int k=1;k<=n+1;++k)  a[j][k] -= a[i][k]*temp;
			}
		}
	}
	for(int i=1;i<=n;++i)	printf("%.2f\n",a[i][n+1]); //最后得到简化阶梯矩阵
	return 0;
}

```

### 筛质数

#### 埃氏筛法

```cpp
#include <iostream>

using namespace std;

const int N = 1000;

int prime[N];
int st[N];
int n, k;

//O(nloglogn)
int getprimes(int n){
    int idx = 1;
    for(int i = 2; i <= n; i ++){
        if(!st[i]){
            prime[idx++] = i;
            for(int j = i + i; j <= n; j += i) st[j] = 1;
        }
    }
    return idx - 1;
}

int main(){
    int t;
    cin >> t;
    while(t --){
        cin >> n >> k;
        int cnt = getprimes(n);
        int res = 0;
        for(int i = 1; i <= cnt; i ++){
            for(int j = 1; j <= i - 2; j ++){
                if(prime[i] == prime[j] + prime[j + 1] + 1){
                    res ++;
                }
            }
        }
        if(res >= k) puts("YES");
        else puts("NO");
    }
    return 0;
}
```

#### 线性筛法

```cpp
#include <iostream>

using namespace std;

const int N = 1e6 + 1;
bool st[N];
int cnt;
int primes[N];

void get_primes(int n){
    for(int i = 2; i <= n; i++){
        if(!st[i]) primes[cnt++] = i; 
        for(int j = 0; primes[j] <= n / i; j++){
            st[primes[j] * i] = true;
            if(i % primes[j] == 0) break;
        }
    }
}

int main(){
    int n; cin >> n;
    get_primes(n);
    cout << cnt;
    return 0;
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

mex(A): 返回A集合中未曾出现过的最小的自然数

sg(x): 如果处于最终态，则返回0；否则返回该节点可以到达的状态（A集合）中未曾出现过的最小的自然数

例题：

https://www.acwing.com/problem/content/895/

```cpp
#include <iostream>
#include <cstring>
#include <unordered_set>

using namespace std;

const int N = 110, M = 10010;

int f[M];
int a[N];
int n, k;

int sg(int x){
    if(f[x] != -1) return f[x];
    unordered_set<int> reach_set;
    for(int i = 1; i <= k; i ++){
        int cut = a[i];
        if(x >= a[i]) reach_set.insert(sg(x - a[i]));
    }
    for(int i = 0; ; i ++){
        if(!reach_set.count(i)) return f[x] = i;
    }
}

int main(){
    memset(f, -1, sizeof f);
    cin >> k;
    for(int i = 1; i <= k; i ++) cin >> a[i];
    cin >> n;
    int s = 0;
    for(int i = 1; i <= n; i ++){
        int x;
        cin >> x;
        s ^= sg(x);
    }
    if(s) puts("Yes");
    else puts("No");
    return 0;
}
```



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

struct Edge
{
    int a, b, w;

    bool operator< (const Edge &W)const
    {
        return w < W.w;
    }
}edges[M];

int find(int x)
{
    if (p[x] != x) p[x] = find(p[x]);    // 路径压缩
    return p[x];
}

int kruskal()
{
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

int main()
{
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

#### 权值线段树

https://www.cnblogs.com/young-children/p/11787493.html

**定义：**

权值线段树，基于普通线段树，但是不同。

举个栗子：对于一个给定的数组，普通线段树可以维护某个子数组中数的和，而权值线段树可以维护某个区间内数组元素出现的次数。

在实现上，由于值域范围通常较大，权值线段树会采用离散化或动态开点的策略优化空间。单次操作时间复杂度o(logn) 

权值线段树的节点用来表示一个区间的数出现的次数 例如： 数 1和2 分别出现3次和5次，则节点1记录 3，节点2 记录5， 1和2的父节点记录它们的和8 .

**存储结构**：

堆式存储：rt ,l, r,   rt<<! , l, m   rt<<1|1 ,m+1, r 

结点式存储 struct Node { int sum ,l , r :};

**基本作用：**

查询第k小或第k大。

查询某个数排名。

查询整干数组的排序。

查询前驱和后继（比某个数小的最大值，比某个数大的最小值）

**基本操作：**

**单点修改 (单个数出现的次数+1)**



```cpp
    void update(int l,int r,int rt,int pos) // 当前区间范围 l r 节点 rt    位置 pos 
    {
        if(l==r) t[rt]++;
        else
        {
            int mid=(l+r)/2;
            if(pos<=mid) add(l,mid,rt*2,pos); else add(mid+1,r,rt*2+1,pos);
            t[rt]=t[rt*2]+t[rt*2+1];
        {
    }
```



查询一个数出现的次数



```cpp
    int query(int l,int r,int rt,int pos)
    {
        if(l==r) return t[rt];
        else
        {
            int mid=(l+r)/2;
            if(pos<=mid) return find(l,mid,rt*2,pos); else return find(mid+1,r,rt*2+1,pos);
        }
    }
```



查询一段区间数出现的次数 查询区间 【x,y]

递归+二分



```cpp
    int query(int l,int r,int rt,int x,int y)
    {
        if(l==x&&r==y) return t[rt];
        else
        {
            int mid=(l+r)/2;
            if(y<=mid) return find(l,mid,rt*2,x,y);
            else if(x>mid) return find(mid+1,r,rt*2+1,x,y);
            else return find(l,mid,rt*2,x,mid)+find(mid+1,r,rt*2+1,mid+1,y);
        }
    }
```



**查询所有数的第k大值**
这是权值线段树的核心，思想如下：
到每个节点时，如果右子树的总和大于等于k kk，说明第k kk大值出现在右子树中，则递归进右子树；否则说明此时的第k kk大值在左子树中，则递归进左子树，注意：此时要将k kk的值减去右子树的总和。
为什么要减去？
如果我们要找的是第7 77大值，右子树总和为4 44，7−4=3 7-4=37−4=3，说明在该节点的第7 77大值在左子树中是第3 33大值。
最后一直递归到只有一个数时，那个数就是答案。

```cpp
    int kth(int l,int r,int rt,int k)
    {
        if(l==r) return l;
        else
        {
            int mid=(l+r)/2,s1=f[rt*2],s2=f[rt*2+1];
            if(k<=s2) return kth(mid+1,r,rt*2+1,k); else return kth(l,mid,rt*2,k-s2);
        }
    }
```

 

模板题：

HDU – 1394

给你一个序列，你可以循环左移，问最小的逆序对是多少？？？

逆序对其实是寻找比这个数小的数字有多少个，这个问题其实正是权值线段树所要解决的

我们把权值线段树的单点作为1-N的数中每个数出现的次数，并维护区间和，然后从1-N的数，在每个位置，查询比这个数小的数字的个数，这就是当前位置的逆序对，然后把当前位置数的出现的次数+1，就能得到答案。

然后我们考虑循环右移。我们每次循环右移，相当于把序列最左边的数字给放到最右边，而位于序列最左边的数字，它对答案的功效仅仅是这个数字大小a[i]-1，因为比这个数字小的数字全部都在它的后面，并且这个数字放到最后了，它对答案的贡献是N-a[i],因为比这个数字大数字全部都在这个数字的前面，所以每当左移一位，对答案的贡献其实就是

Ans=Ans-(a[i]-1)+n-a[i]

由于数字从0开始，我们建树从1开始，我们把所有数字+1即可

```cpp
#include<iostream>
#include<string.h>
#include<algorithm>
#include<stdio.h>
using namespace std;
const int maxx = 5005;
int tree[maxx<<2];
inline int L(int root){return root<<1;};
inline int R(int root){return root<<1|1;};
inline int MID(int l,int r){return (l+r)>>1;};
int a[maxx];
void update(int root,int l,int r,int pos){
   if (l==r){
     tree[root]++;
     return;
   }
   int mid=MID(l,r);
   if (pos<=mid){
      update(L(root),l,mid,pos);
   }else {
      update(R(root),mid+1,r,pos);
   }
   tree[root]=tree[L(root)]+tree[R(root)];
}
int query(int root,int l,int r,int ql,int qr){
     if (ql<=l && r<=qr){
        return tree[root];
     }
     int mid=MID(l,r);
     if (qr<=mid){
        return query(L(root),l,mid,ql,qr);
     }else if (ql>mid){
        return query(R(root),mid+1,r,ql,qr);
     }else {
        return query(L(root),l,mid,ql,qr)+query(R(root),mid+1,r,ql,qr);
     }
}
int main(){
  int n;
  while(~scanf("%d",&n)){
    int ans=0;
    memset(tree,0,sizeof(tree));
    for (int i=1;i<=n;i++){
        scanf("%d",&a[i]);
        a[i]++;
        ans+=query(1,1,n,a[i],n);
        update(1,1,n,a[i]);
    }
    int minn=ans;
    for (int i=1;i<=n;i++){
      ans=ans+(n-a[i]+1)-a[i];
      minn=min(ans,minn);
    }
    printf("%d\n",minn);
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

#### STL unordered_set/map

##### map

(注意c++中map主要是红黑树即平衡树实现，unordered_map才是哈希表实现)

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

#### 例题1：第13届蓝桥杯第7题：

给定一个长度为 $ N $ 的整数序列：$ A_1, A_2, · · · , A_N $。

现在你有一次机会，将其中连续的 $ K $ 个数修改成任意一个相同值。

请你计算如何修改可以使修改后的数列的最长不下降子序列最长，请输出这个最长的长度。

最长不下降子序列是指序列中的一个子序列，子序列中的每个数不小于在它之前的数。

输入格式

输入第一行包含两个整数 $ N $ 和 $ K $。

第二行包含 $ N $ 个整数 $ A_1, A_2, · · · , A_N $。

输出格式

输出一行包含一个整数表示答案。

数据范围

对于 $ 20\% $ 的评测用例，$ 1 ≤ K ≤ N ≤ 100 $；  
对于 $ 30\% $ 的评测用例，$ 1 ≤ K ≤ N ≤ 1000 $；  
对于 $ 50\% $ 的评测用例，$ 1 ≤ K ≤ N ≤ 10000 $；  
对于所有评测用例，$ 1 ≤ K ≤ N ≤ 10^5 $，$ 1 ≤ A_i ≤ 10^6 $。

输入样例：

```
5 1
1 4 2 8 5
```

输出样例：

```
4
```

仅附大佬题解，因为我也还不是完全理解

__佬的：__

以dp1[i]表示从前往后以a[i]结尾的最长不下降子序列的长度

以dp2[i]表示从前往后以a[i]开头的最长不下降子序列的长度

我们最后的答案显然会由三部分组成

如果第一部分为dp1[i]，第二部分就是k，第三部分就是max(dp2[j]),i+k+1≤j≤n且a[j]≥a[i]

第三部分我们可以用线段树快速得到

处理dp1和dp2的过程就是一个普通的求最长不下降子序列的过程

二分，权值树状数组，权值线段树，都是O(nlogn)的复杂度，随便选一个即可

如果写权值线段树的话，要先离散化一下，不然可能炸空间

> 作者：Goku74
> 链接：https://www.acwing.com/solution/content/143797/
> 来源：AcWing
> 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

代码：

```cpp
#include <bits/stdc++.h>
const int N = 1e5 + 10;
int n, m, k, ans;
int a[N];
int b[N];       // 用于离散化的数组
int dp1[N];     // dp1[i]表示从前往后以a[i]结尾的最长不下降子序列的长度
int dp2[N];     // dp2[i]表示从前往后以a[i]开头的最长不下降子序列的长度
int find(int x) //返回整数a[i]在b数组中的下标
{
    int l = 1, r = m;
    while (l < r)
    {
        int mid = l + r >> 1;
        if (b[mid] >= x)
            r = mid;
        else
            l = mid + 1;
    }
    return l;
}
struct
{
    int maxv;
} seg[N * 4];
void pushup(int id)
{
    seg[id].maxv = std::max(seg[id << 1].maxv, seg[id << 1 | 1].maxv);
}
// 建树时l和r是虚拟的不存在于节点上的，最小节点存的是以a[i]结尾的最长不下降子序列的长度
void build(int id, int l, int r)
{
    if (l == r)
        seg[id].maxv = 0;
    else
    {
        int mid = l + r >> 1;
        build(id << 1, l, mid);
        build(id << 1 | 1, mid + 1, r);
        pushup(id);
    }
}
void change(int id, int l, int r, int pos, int val)
{
    if (l == r)
        seg[id].maxv = std::max(seg[id].maxv, val);
    else
    {
        int mid = l + r >> 1;
        if (pos <= mid)
            change(id << 1, l, mid, pos, val);
        else
            change(id << 1 | 1, mid + 1, r, pos, val);
        pushup(id);
    }
}
int query(int id, int l, int r, int ql, int qr)
{
    if (l == ql && r == qr)
        return seg[id].maxv;
    int mid = l + r >> 1;
    if (qr <= mid)
        return query(id << 1, l, mid, ql, qr);
    else if (ql >= mid + 1)
        return query(id << 1 | 1, mid + 1, r, ql, qr);
    else
        return std::max(query(id << 1, l, mid, ql, mid), query(id << 1 | 1, mid + 1, r, mid + 1, qr));
}
signed main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);
    std::cin >> n >> k;
    for (int i = 1; i <= n; ++i)
        std::cin >> a[i], b[i] = a[i];
    std::sort(b + 1, b + n + 1); //排序
    if (n == k)
    {
        std::cout << n << '\n';
        return 0;
    } 
    m = 1;
    for (int i = 2; i <= n; i++) //去重
        if (b[i] != b[m])
            b[++m] = b[i];
    for (int i = 1; i <= n; ++i)
        a[i] = find(a[i]);
    build(1, 1, m); //建权值线段树
    for (int i = 1; i <= n - k; ++i)
    {
        //先求出以a[i]结尾的最长不下降子序列的长度
        //用以1结尾的一直到以a[i]结尾的最大值
        //这里其实是在枚举倒数第2个数是啥
        dp1[i] = query(1, 1, m, 1, a[i]) + 1;
       //再把求得的dp[i]的值存下来
        change(1, 1, m, a[i], dp1[i]);
    }
    build(1, 1, m); // dp1已经处理完，重新建树处理dp2
    for (int i = n; i >= k + 1; --i)
    {
        ans = std::max(ans, dp1[i - k] + k + query(1, 1, m, a[i - k], m));
        // 第一段为dp1[i-k]，第二段为k，第三段为max(dp2[j]),i+k+1<=j<=n且a[j]>=a[i]
        dp2[i] = query(1, 1, m, a[i], m) + 1;
        change(1, 1, m, a[i], dp2[i]);
    }
    // 特殊情况
    for (int i = 1; i <= n - k; ++i)
        ans = std::max(ans, dp1[i] + k);
    for (int i = n; i >= k + 1; --i)
        ans = std::max(ans, dp2[i] + k);
    std::cout << ans << '\n';
    return 0;
}
```

### 背包问题

鄙人才疏学浅，此中鄙陋甚多，望海涵。

#### 01背包

先简单的描述一下题目：就是在一定体积限制下选物品放入背包中，使其价值达到最大。（AcWing 2. 01背包问题）

特征：一个物品最多只能选一次。
思路：
{
状态表示：f\[i][j]表示从前i个物品里选择（一个物品最多选一次），总体积不超过j的价值；
属性：最大值
状态计算：<font color=red>```f[i][j]=max(f[i-1][j],f[i-1][j-w[i]]+v[i])```</font>(解释；表示从前i个物品里选择体积不超过j的物品价值的最大值是从要么不选这个物品，和要么选择一个这个物品的价值之间进行选择)
}

##### 01背包问题的二维代码

C++ 代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=1010;

int n,m;
int f[N][N],w[N],v[N];

int main()
{
    cin>>n>>m;
    //动态规划问题一般下标从1开始，便于从0的情况递推结果。
    for(int i=1;i<=n;i++) cin>>w[i]>>v[i];

    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            f[i][j]=f[i-1][j];
            if(j>=w[i])
                f[i][j]=max(f[i][j],f[i-1][j-w[i]]+v[i]);
        }
    
    cout<< f[n][m] <<endl;
    return 0;

}
```

显然不优化到一维还是不能满足我们追求完美的精神，一维优化的思路：我们先尝试着把f[i][j]中[i]这一维去掉，对于<font color=red>```if(j>=w[i])```</font>,我们只需要让j时刻大于w[i]即可，但主要的矛盾是在<font color=red>`f[i][j]=max(f[i][j],f[i-1][j-w[i]]+v[i])`</font>中，我们去掉了[i-1]这一维，由于j始终是从小到大枚举，便不能保证它是从上一次更新好的状态转移过来的，会实现自我更新（顺带一提，j的枚举正是01背包与完全背包区别的关键，当j从小到大枚举时便能实现自我更新，摆脱了物品个数这一限制），因此当我们将j从大到小枚举时f[j]就不会实现自我更新，保证一个物品只能选一次的这个限制）

##### 01背包一维优化

C++ 代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=1010;

int f[N];
int w[N],v[N];
int n,m;

int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>w[i]>>v[i];

    for(int i=1;i<=n;i++)
        for(int j=m;j>=w[i];j--)
        f[j]=max(f[j],f[j-w[i]]+v[i]);
    
    cout<<f[m]<<endl;
    return 0;

}


```

#### 完全背包问题

先简单的描述一下题目：就是在一定体积限制下选物品放入背包中，使其价值达到最大。（AcWing 3. 完全背包问题）

特征：一个物品可以选无数次。
思路：
{
状态表示：f[i][j]表示从前i个物品里选择（一个物品可以选多次），总体积不超过j的价值；
属性：最大值
状态计算：<font color=red>`f[i][j]=max(f[i][j],f[i-1][j-k*w[i]]+k*v[i])`</font>(解释；表示从前i个物品里选择体积不超过j的物品价值的最大值是从要么不选这个物品，和要么选择K个这个物品的价值之间进行选择，其中当k=0时就是f\[i-1\][j]的情况)
}

##### 完全背包问题的三重循环代码

C++ 代码

```cpp
#include<iostream>

using namespace std;

const int N=1010;

int f[N][N];
int w[N],v[N];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++)   cin>>w[i]>>v[i];

    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            for(int k=0;k*w[i]<=j;k++)
                f[i][j]=max(f[i][j],f[i-1][j-k*w[i]]+k*v[i]);
    
    cout<<f[n][m]<<endl;
    return 0;

}
```

接下来我们应该考虑二重循环二维优化了，毕竟o(n3)的时间复杂度太高，极容易被卡(百分之99哦)。



##### 二重循环二维的优化过程

```cpp
f[i][j]     =max(f[i][j],f[i-1][j-k*w[i]]+k*v[i]);
f[i][j]     =max(f[i-1][j],f[i-1][j-w[i]]+v[i],f[i-1][j-2*w[i]]+2*v[i],.....)
f[i][j-w[i]]=max(          f[i-1][j-w[i]],     f[i-1][j-2*w[i]]+1*v[i],.....)
```

我们不难发现，把<font color=red>`f[i][j]`</font>展开会发现上面的规律。
因此凭借这个我们就可以消灭掉k这个大麻烦了，因为<font color=red>`f[i][j]=max(f[i-1][j],f[i][j-w[i]]+v[i]);`</font>



C++代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=1010;


int f[N][N];
int n,m;
int w[N],v[N];

int main()
{
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>w[i]>>v[i];

    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            f[i][j]=f[i-1][j];
            if(j>=w[i])
            f[i][j]=max(f[i][j],f[i][j-w[i]]+v[i]);
        }
    
    cout<< f[n][m] <<endl;
    return 0;

}
```


引用一下上面01背包二维的代码

```cpp
for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            f[i][j]=f[i-1][j];
            if(j>=w[i])
                f[i][j]=max(f[i][j],f[i-1][j-w[i]]+v[i]);
        }
```

再引用一下完全背包的二维代码

```cpp
 for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            f[i][j]=f[i-1][j];
            if(j>=w[i])
            f[i][j]=max(f[i][j],f[i][j-w[i]]+v[i]);
        }
```

写到这里，你就会惊奇的发现这个代码和01背包真的只有一点点不一样啊！聪明的同学们已经观察到了就是
<font color=red>`f[i][j]=max(f[i][j],f[i-1][j-w[i]]+v[i]);`</font> 和 <font color=red>`f[i][j]=max(f[i][j],f[i][j-w[i]]+v[i]);`</font>的不同
不难想到之前提过的能否更新自己的区别，也就是更新源于本层（指i）还是上一层（i-1），我们之前也说过这也就是为什么一维优化后01背包和完全背包一个是正枚举j ，一个是倒枚举j的原因了。

##### 完全背包一维最终优化(虽然经历了千辛万苦)

C++代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=1010;

int f[N],w[N],v[N];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++)   scanf("%d%d",&w[i],&v[i]);
    for(int i=1;i<=n;i++)
        for(int j=w[i];j<=m;j++)
            f[j]=max(f[j],f[j-w[i]]+v[i]);

    cout<< f[m] <<endl;
    return 0;

}
```

#### 多重背包1

先简单的描述一下题目：就是在一定体积限制下选物品放入背包中，使其价值达到最大。（AcWing 3. 完全背包问题）

特征：一个物品有严格的次数限制。
思路：
{
状态表示：f\[i][j]表示从前i个物品里选择（一个物品有严格的次数限制），总体积不超过j的价值；
属性：最大值
状态计算：<font color=red>`f[i][j]=max(f[i-1][j],f[i-1][j-k*w[i]]+k*v[i])`</font>(解释；表示从前i个物品里选择体积不超过j的物品价值的最大值是从要么不选这个物品，和要么选择K个这个物品的价值之间进行选择)
}

这里不难发现其实朴素版的多重背包思路和三维的完全背包问题思路几乎一致，其实也就多了一个限制条件而已（就是对k的大小的限制），这里就不再赘述了，直接上代码！

##### 多重背包问题的三重循环代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=110;

int f[N][N],s[N],w[N],v[N];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>w[i]>>v[i]>>s[i];

    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            for(int k=0;k<=s[i] && k*w[i]<=j;k++)
                f[i][j]=max(f[i][j],f[i-1][j-k*w[i]]+k*v[i]);
    
    cout<< f[n][m] <<endl;
    return 0;

}
```

实际上也就多了<font color=red>`k<=s[i]`</font>这个限制而已。
那么我们来想想既然上面的问题都可以一维优化，那么这个多重背包问题能不能优化为一维呢（其实在这里这个优化实用性并不大，因为多数情况下多重背包朴素版的时间复杂度太高，一般都会被卡掉，而且优化为一维时间复杂度并不会减少，但是为了追求完美，就必须。。。）我们观察一下多重背包问题的状态转移方程
<font color=red>`f[i][j]=max(f[i][j],f[i-1][j-k*w[i]]+k*v[i]);`</font>当我们去掉[i]这一维时，又产生了似曾相识的问题，没错，就是我们在优化01背包时，保证不能自我更新，需要凭借上一层的更新结果来更新本层，聪明的同学一定想到了，既然矛盾相似，那么优化的结果应该也相似，没错，结果就是将j 倒枚举。我们先给出代码，再验证会不会仍然存在矛盾。

##### 多重背包问题一维优化代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=110;

int f[N],s[N],w[N],v[N];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++) cin>>w[i]>>v[i]>>s[i];

    for(int i=1;i<=n;i++)
        for(int j=m;j>=w[i];j--)
            for(int k=0;k<=s[i] && k*w[i]<=j;k++)
                f[j]=max(f[j],f[j-k*w[i]]+k*v[i]);
    
    cout<< f[m] <<endl;
    return 0;

}
```

结合01背包来理解其实不存在矛盾（还不懂的同学可以再回去研究一下01背包的优化过程）

##### 多重背包问题打包版

问题和上一个多重背包问题一样，只不过在这里数据范围变大了许多（AcWing 5. 多重背包问题 II）
思路:
{
既然三重循环时间复杂度怎么高，那我们能不能想一种避免三重循环的方法呢？答案是肯定的！
这里我们要介绍一个二进制数的重要性质 就是任何数N都能由它的2的从0 到 logN 次方加和可得 （最后一个不够二的整次方数补齐即可），那么这样的话，我们就可以将一个有严格次数限制的物品打包成为只能使用一次的小包，对应的小包的体积和价值都要发生变化（与小包中物品个数有关）

状态表示和状态计算同01背包。
}

##### 多重背包最终优化代码

C++代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=11000,M=2010;
//这里N的大小是1000（log1000 + 1） 计算出来的。

int w[N],v[N],cnt;
int f[M];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        int a,b,c;
        scanf("%d%d%d",&a,&b,&c);
        int t=1;
        while(c)
        {
           if(c>=t)
           {
               cnt++;
               w[cnt]=a*t;
               v[cnt]=b*t;
               c-=t;
               t*=2;
           }
           else if(c)
           {
               cnt++;
               w[cnt]=a*c;
               v[cnt]=b*c;
               break;
           }
        }
    }
    for(int i=1;i<=cnt;i++)
        for(int j=m;j>=w[i];j--)
            f[j]=max(f[j],f[j-w[i]]+v[i]);

    cout<< f[m] <<endl;
    return 0;

}


```

#### 分组背包问题

先简单的描述一下题目：就是在一定体积限制下选物品放入背包中，使其价值达到最大。（AcWing 9. 分组背包问题）

特征：一组物品只能选择其中的一个。
思路：
{
状态表示：f\[i][j]表示从前i组物品里选择(一组物品只能选择其中的一个)，总体积不超过j的价值；
属性：最大值
状态计算：<font color=red>`f[i][j]=max(f[i-1][j],f[i-1][j-w[i][k]]+v[i][k])`</font>(解释；表示从前i组物品里选择体积不超过j的物品价值的最大值是从要么不选这个物品，和要么选择这个组中第k个物品的价值之间进行选择)
}
其实这个问题更像是套着多重背包外套的01背包问题，其实不难想到，我们只需要在二维01背包的基础上再枚举一个中间量k即可（k表示在这个组中的第k个物品），这里也不再赘述了，直接上代码。

##### 分组背包问题一维优化代码

```cpp
#include<iostream>
#include<algorithm>

using namespace std;

const int N=110;

int w[N][N],v[N][N];
int f[N],s[N];

int main()
{
    int n,m;
    cin>>n>>m;
    for(int i=1;i<=n;i++)
    {
        cin>>s[i];
        for(int j=1;j<=s[i];j++)
            cin>>w[i][j]>>v[i][j];
    }
    for(int i=1;i<=n;i++)
        for(int j=m;j>=1;j--)
            for(int k=1;k<=s[i];k++)
                if(w[i][k]<=j)
                    f[j]=max(f[j],f[j-w[i][k]]+v[i][k]);

    cout<< f[m] <<endl;
    return 0;

}
```

我本人觉得这个是上述问题最好理解的一个问题（哈哈）

我们再总结一下，这五类的背包问题都可以写成一维优化的形式，这其中只有完全背包是正枚举！！！

作者：Accepting
		链接：https://www.acwing.com/blog/content/1992/
		来源：AcWing
		著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

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

# 时下流行

## Sunday字符串匹配

```cpp
#include<bits/stdc++.h>
using namespace std;
 
string a;
string b;
int nextn[256];
 
void init(void)
{
    memset(nextn,-1,sizeof(nextn));
    int len = b.size();
    for(int i = 0 ; i < len; i++) nextn[b[i]] = i;
}
 
int sunday(void)
{
    init();
    int alen = a.size();
    int blen = b.size();
    if(alen==0) return -1;
    for(int i = 0 ; i <= alen-blen;)
    {
        int j = i;
        int k = 0;
        for(; a[j]==b[k] && j<alen && k<blen; j++,k++);
        if(k==blen) return i;
        else
        {
            if(i+blen<alen) i+=(blen-nextn[a[i+blen]]);
            else return -1;
            }
    }
    return -1;
}
int main(void)
{
    cin>>a;
    cin>>b;
    cout<<sunday()<<endl;
    return 0;
}
————————————————
版权声明：本文为CSDN博主「两米长弦」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/peng0614/article/details/79460863
```

