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

using namespace std;

using LL = long long;
using PII = pair<int, int>;
using PLL = pair<LL, LL>;
using PDI = pair<double, int>;
using PDD = pair<double, double>;
using ULL = unsigned long long;

const int mod = 998244353;
const int N = 1e5 + 10, M = 2e5 + 10;
const int INF = (1LL << 31) - 1;
const LL LLF = 2e18 - 1;

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

// 读入优化
inline int read() {
    int s = 0, w = 1;
    char c = getchar();
    while (c < 48 || c > 57) {
        if (c == '-') w = -1;
        c = getchar();
    }
    while (c >= 48 && c <= 57)
        s = (s << 1) + (s << 3) + c - 48, c = getchar();
    return s * w;
}

// 输出优化
inline void pf(LL x) {
    if (x < 0) putchar('-'), x = -x;
    if (x > 9) pf(x / 10);
    putchar(x % 10 + 48);
}

// 求组合数
const int Z = 1e6 + 10;
int fc[Z], ifc[Z];
int bin(int n, int m) {
  if(n < m) return 0;
  return 1ll * fc[n] * ifc[m] % mod * ifc[n - m] % mod;
}
void init_fac(int Z) {
  for(int i = fc[0] = 1; i < Z; i++) fc[i] = 1ll * fc[i - 1] * i % mod;
  ifc[Z - 1] = qpow(fc[Z - 1], mod - 2);
  for(int i = Z - 2; ~i; i--) ifc[i] = 1ll * ifc[i + 1] * (i + 1) % mod;
}

// 线段树
/*
struct Node {
    int l, r, sum, tag;
}tr[N * 4];
int planted, plucked;

void pushup(int u) {
    tr[u].sum = tr[u << 1].sum + tr[u << 1 | 1].sum;
}

void pushdown(int u) {
    if (tr[u].tag == -1) return;
    if (tr[u].l == tr[u].r) return;
    tr[u << 1].sum = tr[u].tag * tr[u << 1].sum;
    tr[u << 1 | 1].sum = tr[u].tag * tr[u << 1 | 1].sum;
    tr[u << 1].tag = tr[u].tag;
    tr[u << 1 | 1].tag = tr[u].tag;
    tr[u].tag = -1;
}

void build(int u, int l, int r) {
    tr[u] = {l, r, 0, 1};
    if (l == r) {
        tr[u].sum = 1;
        return;
    }
    int mid = l + r >> 1;
    build(u << 1, l, mid), build(u << 1 | 1, mid + 1, r);
    pushup(u);
}

void add(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        if (tr[u].tag == 1 || tr[u].tag == 2) {
            return;
        }
        if (tr[u].tag == 0) {
            tr[u].sum = tr[u].r - tr[u].l + 1;
            tr[u].tag = 2;
            planted += tr[u].r - tr[u].l + 1;
            return;
        }
        if (tr[u].tag == -1) {
            add(u << 1, l, r);
            add(u << 1 | 1, l, r);
            return;
        }
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (mid >= l) add(u << 1, l, r);
    if (mid < r) add(u << 1 | 1, l, r);
    pushup(u);
}

void remove(int u, int l, int r) {
    if (tr[u].l >= l && tr[u].r <= r) {
        if (tr[u].tag == 0) {
            return;
        }
        if (tr[u].tag == 1) {
            tr[u].sum = 0;
            tr[u].tag = 0;
            return;
        }
        if (tr[u].tag == 2) {
            tr[u].sum = 0;
            tr[u].tag = 0;
            plucked += tr[u].r - tr[u].l + 1;
            return;
        }
        if (tr[u].tag == -1) {
            remove(u << 1, l, r);
            remove(u << 1 | 1, l, r);
            return;
        }
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (mid >= l) remove(u << 1, l, r);
    if (mid < r) remove(u << 1 | 1, l, r);
    pushup(u);
}

int query(int u, int l, int r) {
    int res = 0;
    if (tr[u].l >= l && tr[u].r <= r) {
        return tr[u].sum;
    }
    pushdown(u);
    int mid = tr[u].l + tr[u].r >> 1;
    if (mid >= l) res += query(u << 1, l, r);
    if (mid < r) res += query(u << 1 | 1, l, r);
    pushup(u);
    return res;
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
    fprintf(stderr, "%.3lf MB\n", (&Mbe - &Med) / 1048576.0);
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    int T = 1;
    cin >> T;
    while (T --) solve();
    cerr << 1e3 * clock() / CLOCKS_PER_SEC << " ms\n";
    return 0;
}
