#include <iostream>
#include <algorithm>
#include <random>
#include <functional>
#include <queue>
#include <stack>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>

// #pragma O(2)
// #define int long long
// 返回x的二进制中1的个数
#define ppc(x) __builtin_popcount(x)
// 返回x的二进制中前导0的个数
#define clz(x) __builtin_clz(x)
// 返回x的二进制中后导0的个数
#define ctz(x) __builtin_ctz(x)
// lg(8) = 3, lg(16) = 4, 计算x的以2为底的对数的向下取整, 即x的最高位1的位置
#define lg(x) __lg(x)

using namespace __gnu_pbds;
using namespace std;

using LL = long long;
using PII = pair<int, int>;
using PLL = pair<LL, LL>;
using PDI = pair<double, int>;
using PDD = pair<double, double>;
using ULL = unsigned long long;
template<class T> using sset = tree<T, null_type, less<>, rb_tree_tag, tree_order_statistics_node_update>;

const int mod = 998244353;
const int N = 1e5 + 10, M = 2e5 + 10;
const int INF = 0x3f3f3f3f;
const LL LLF = 0x3f3f3f3f3f3f3f3f;

// 用于检测程序占用内存大小
bool Mbe;

// 随机数生成器
// mt19937 rnd(chrono::steady_clock::now().time_since_epoch().count());
mt19937_64 rnd(1064);
int rd(int l, int r) {return rnd() % (r - l + 1) + l;}

// 取模运算
void addt(int &x, int y) {x += y, x >= mod && (x -= mod);}
void subt(int &x, int y) {x -= y, x < 0 && (x += mod);}
void mult(int &x, int y) {x = (LL)x * y % mod;}
int add(int x, int y) {return x + y >= mod ? x + y - mod : x + y;}
int sub(int x, int y) {return x - y < 0 ? x - y + mod : x - y;}
int mul(int x, int y) {return (LL)x * y % mod;}

// 快速幂
int qpow(int a, int b) {
    int res = 1;
    while (b) {
        if (b & 1) mult(res, a);
        mult(a, a), b >>= 1;
    }
    return res;
}

// 最大公约数
int gcd(int a, int b) {return b ? gcd(b, a % b) : a;}

// 逆元
int inv(int x) {return qpow(x, mod - 2);}

// 二分
int bsearch(int l, int r, function<bool(int)> f) {
    while (l < r) {
        int mid = (l + r) >> 1;
        if (f(mid)) r = mid;
        else l = mid + 1;
    }
    return l;
}

void setIO() {
    ios_base::sync_with_stdio(0);
    cin.tie(0);
}

// 读入优化
template<typename T>
inline T read(){
	T a = 0;
    bool s = 0;
	char ch = getchar();
	while (ch > '9' || ch < '0'){
		if (ch == '-') s ^= 1;
		ch = getchar();
	}
	while (ch >= '0' && ch <= '9'){
		a = (a << 3) + (a << 1) + (ch ^ 48);
		ch = getchar();
	}
	return s ? -a : a;
}

// 输出优化
inline void pf(LL x) {
    if (x < 0) putchar('-'), x = -x;
    if (x > 9) pf(x / 10);
    putchar(x % 10 + 48);
}

/*
// 求组合数
const int Z = 1e6 + 10;
int fc[Z], ifc[Z];
int bin(int n, int m) {
    if (n < m) return 0;
    return 1ll * fc[n] * ifc[m] % mod * ifc[n - m] % mod;
}
void init_fac(int Z) {
    for (int i = fc[0] = 1; i < Z; i++) fc[i] = 1ll * fc[i - 1] * i % mod;
    ifc[Z - 1] = qpow(fc[Z - 1], mod - 2);
    for (int i = Z - 2; ~i; i--) ifc[i] = 1ll * ifc[i + 1] * (i + 1) % mod;
}
*/

// 平衡树
/*
struct Node {
    int s[2], p, v;
    int size, flag;
    void init(int _v, int _p) {
        v = _v, p = _p;
        size = 1;
    }
}tr[N];
int root, idx;

void pushup(int x) {
    tr[x].size = tr[tr[x].s[0]].size + tr[tr[x].s[1]].size + 1;
}

void pushdown(int x) {
    if (tr[x].flag) {
        swap(tr[x].s[0], tr[x].s[1]);
        tr[tr[x].s[0]].flag ^= 1;
        tr[tr[x].s[1]].flag ^= 1;
        tr[x].flag = 0;
    }
}

void rotate(int x) {
    int y = tr[x].p, z = tr[y].p;
    int k = tr[y].s[1] == x;
    tr[z].s[tr[z].s[1] == y] = x, tr[x].p = z;
    tr[y].s[k] = tr[x].s[k ^ 1], tr[tr[x].s[k ^ 1]].p = y;
    tr[x].s[k ^ 1] = y, tr[y].p = x;
    pushup(y), pushup(x);
}

void splay(int x, int k) {
    while (tr[x].p != k) {
        int y = tr[x].p, z = tr[y].p;
        if (z != k) {
            if ((tr[y].s[1] == x) ^ (tr[z].s[1] == y)) rotate(x);
            else rotate(y);
        }
        rotate(x);
    }
    if (!k) root = x;
}

void insert(int v) {
    int u = root, p = 0;
    while (u) p = u, u = tr[u].s[v > tr[u].v];
    u = ++ idx;
    if (p) tr[p].s[v > tr[p].v] = u;
    tr[u].init(v, p);
    splay(u, 0);
}

int get_k(int k) {
    int u = root;
    while (true) {
        pushdown(u);
        if (tr[tr[u].s[0]].size >= k) u = tr[u].s[0];
        else if (tr[tr[u].s[0]].size + 1 == k) return u;
        else k -= tr[tr[u].s[0]].size + 1, u = tr[u].s[1];
    }
    return -1;
}

void output(int u) {
    pushdown(u);
    if (tr[u].s[0]) output(tr[u].s[0]);
    if (tr[u].v >= 1 && tr[u].v <= n) printf("%d ", tr[u].v);
    if (tr[u].s[1]) output(tr[u].s[1]);
}
*/

/* template above */

void solve() {

}

bool Med;
int main() {
    fprintf(stderr, "%.3lf MB\n", (&Med - &Mbe) / 1048576.0);
    // setIO();
    int T = 1;
    cin >> T;
    while (T --) solve();
    cerr << 1e3 * clock() / CLOCKS_PER_SEC << " ms\n";
    return 0;
}
