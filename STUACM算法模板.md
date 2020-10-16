<center><h1 style="color:red">STUACM Algorithm</h1></center>

[TOC]

# 排序

## 选择排序

> 如其名,像冒泡一样每趟把大的上升到最后
>
> 是**稳定**的平均时间为$O(n^2)$,最优为$O(n)$,空间为$O(1)$的排序算法

```c++
const int n;
vector<int> arr(n);	//待排序数组
void bubble_sort()
{
    bool flag=true;	//判断是否已经有序
    for(int i=n-1;i>0&&flag;i--)	//i作为标记,上限是处理n-1次
    {
        flag=false;		//更新标记
        for(int j=0;j<i;j++)
            if(arr[j]>arr[j+1])		//交换逆序对
            {
                flag=true;
                swap(arr[j],arr[j+1]);
            }
    }
}
```

## 插入排序

> 数列前面部分看作有序,依次将后面的元素**逆序**比较插入到前面的有序数列中,数据有序程度越高就越高效
>
> 是**稳定**的平均时间为$O(n^2)$,最优为$O(n)$,空间为$O(1)$的排序算法

```c++
const int n;
vector<int> arr(n);	//待排序数组
void insert_sort()
{
    for(int i=1;i<n;i++)
        for(int j=i;j>0;j--)
            if(arr[j]<arr[j-1])swap(arr[j],arr[j-1]);
    		else break;		//插入完成
}
```

## 希尔排序

> 是插入排序的一个变体,通过不断跳跃划分小的组进行插入排序,使得总体上趋向有序
>
> 其时间复杂度与选用的**增量序列**有关,折半普遍认为时间平均是$O(n^{1.5})$,空间和时间最好是$O(n)$,是**不稳定**的

```c++
const int n;
vector<int> arr(n);	//待排序数组
void shell_sort()
{
    for(int gap=n/2;gap>0;gap/=2)	//减半增量
        for(int i=gap;i<n;i++)		//对于每一个都进行分组插入
            for(int j=i;j>0;j-=gap)		//同一组的进行插入排序
                if (j - gap >= 0 && arr[j] < arr[j - gap])swap(arr[j], arr[j - gap]);
                else break;		//插入完成
}
```

## 快速排序

### 数组

> 选择一个基准点,使得基准点左小右大,再对左右区间递归调用
>
> **平均**时间复杂度为$O(nlog_2n)$,最差的情况要到$O(n^2)$,有$O(log_2n)$的递归空间消耗

```c++
const int n;
vector<int> arr(n);	//待排序数组
void quick_sort(int left,int right)
{
    if(left>=right)return;
    int i=left-1,j=right+1;		//左右两个哨兵
    int base=arr[(left+right)>>1];	//选取区间中间点为基准点
    while(i<j)
    {
        while(arr[++i]<base);	//左哨兵先移动找到第一个比基准点大的数
        while(arr[--j]>base);	//右哨兵移动找到第一个比基准点小的数
        if(i<j)swap(arr[i],arr[j]);	//交换这两个数 保证左边比基准点小,右边比基准点大
    }
    quick_sort(left,j);	//处理左边的		此处一定要是j和j+1
    quick_sort(j+1,right);	//处理右边的
}
```

### 单链表

```c++
typedef struct{		//链表结构
    int val;
    node *next;
}node;
void quick_sort_list(node* left,node *right=nullptr)
{
    if(left==nullptr||left==right)return;
    node *i=left;		//左哨兵
    node *j=left->next;	//右哨兵
    while(j!=right){
        if(j->val<left->val){		//右哨兵遇到了一个基准点小的  应该放在左哨兵的左边
            i=i->next;	//进行扩容
           	swap(i->val,j->val);
        }
        j=j->next;	//进行扩容
    }
    swap(left->val,i->val);	//基准点进行交换
    quick_sort_list(left,i);	//处理左边的
    quick_sort_list(i->next,right);		//处理右边的
}
```

## 堆排序

### 堆

> + 是一棵完全二叉树
> + 树根永远为最大/最小
> + 子树也是堆

```cpp
#define parent (root>>1)
#define left (root<<1)
#define right (left|1)
template<typename T>
class minHeap {
public:
	int size;
	minHeap()
	{
		size = 0;
		T empty;
		v.push_back(empty);
	}
	void push(const T &elem)
	{
		size++;
		v.push_back(elem);
		int root = size;
		while (parent&&v[root] < v[parent])
		{
			swap(v[root], v[parent]);
			root = parent;
		}
	}
	void pop()
	{
		swap(v[1], v[size]);
		v.pop_back();
		size--;
		int root = 1;
		while (left<=size)
		{
			int now = left;
			if (right <= size && v[right] < v[left])now = right;
			if (v[root] < v[now])break;
			swap(v[root], v[now]);
			root = now;
		}
	}
	T top() const
	{
		return v[1];
	}
	void print() const {
		for (int i = 1; i <= size; i++)cout << v[i] << " ";
	}
private:
	vector<T> v;
};
```

### 排序

>利用堆这一种数据结构,把无序数组调整成堆,然后以堆顶和堆尾的交换来达到排序的目的
>
>是**不稳定**的时空复杂度固定为$O(nlog_2n)$的排序算法

```c++
//以下是构建大顶堆 所以排序好的是升序的
const int n;
vector<int> arr(n+1);	//待排序数组	下标从1开始
void heap_adjust(int root,int end)	//root代表当前的需要调整的节点 root+1到end 都已调整好
{
    int top=arr[root];		//当前子堆的顶端
    for(int i=2*root;i<=end;i*=2)	//更新出每个节点的左孩子
    {
        if(i<end&&arr[i]<arr[i+1])i++;	//如果左孩子比右孩子小,则选择右孩子		
        if(top>=arr[i])break;	//如果最开始的顶端比当前要大,证明已找到合适的位置插入
        arr[root]=arr[i];		//把i位置的较大值升上顶
        root=i;	//更新下当前调整到的位置
    }
    arr[root]=top;		//最后的插入
}
void build_heap()
{
    for(int i=n/2;i>0;i--)heap_adjust(i,n);	//先从后向前构建成大顶堆
}
void heap_sort()
{
	build_heap();
    //已经建堆完成
    for(int i=n;i>1;i--){
        swap(arr[1],arr[i]);	//将堆顶和堆尾交换
        heap_adjust(1,i-1);		//把交换之后的重新调整
    }
}
```

## 归并排序

> 把序列向下不断拆分成小的,直到只剩一个,然后向上把两个小的**有序**区间合并成一个大的
>
> 是**不稳定**的时空复杂度固定为$O(nlog_2n)$的排序算法

```c++
const int n;
vector<int> arr(n);	//待排序数组
void merge(int L,int mid,int R)
{
    vector<int> tmp(R-L+1,0);	//开辟额外临时数组存放和并的内容
    int p1=L,p2=mid+1,p=0;		//设置两个指针进行比较
    while(p<R-L+1){
        if(p2>R||(p1<=mid&&arr[p1]<=arr[p2]))tmp[p++]=arr[p1++];	//当区间2用完或者p1所指的不大于p2所指的
        else	//区间1用完了或者p1所指的大于p2所指的
		{
			tmp[p++] = arr[p2++];
			//if (p1 <= mid)cnt+=mid-p1+1;	//在归并排序中计算逆序对的数量
		}
    }
    for(int i=0;i<R-L+1;i++)arr[i+L]=tmp[i];	//还原排序回原数组
}
void merge_sort(int L,int R)
{
    if(L<R)
    {
        int mid=(L+R)>>1;
        merge_sort(L,mid);	//分成左小部分
        merge_sort(mid+1,R);	//分成右小部分
        merge(L,mid,R);		//进行两个小部分的和并
    }
}
```

## 计数排序

> 非比较排序,空间换时间,需要开取数组中$max-min$这么大的空间
>
> 时间复杂度是$O(n+k)$,计数排序可以看作是**稳定**的

```c++
const int n,max_num,min_num;
vector<int> arr(n);	//待排序数组
vector<int< cnt(max_num-min_num+1);
void count_sort()
{
    for(int i=0;i<n;i++)
        cnt[arr[i]-min_num]++;		//进行计数
    for(int i=0,j=0;i<=max_num-min_num&&j<n;i++)	//把计数的结果按顺序安排到原数组中
    	while(cnt[i]--)arr[j++]=i+min_num;
}
```

# 动态规划

## 01背包

### 朴素

> 有$N$件物品和一个容量为$V$的背包,放入第$i$件物品耗费的费用是$C_i$,得到的价值是$W_i$.求解将哪些物品装入背包可以使价值总和最大.

> 定义状态:	$max\_value_{i,v}$为前$i$件物品放入容量为$v$的背包中的最大价值.
>
> 状态转移方程为:	$max\_value_{i,v}=max(max\_value_{i-1,v},max\_value_{i-1,v-C_i}+W_i)$
>
> 意思是考虑当前物品选($max\_value_{i-1,v-C_i}+W_i$)和不选($max\_value_{i-1,v}$)的最优方案

```c++
int N,V;	//物品数和总容量
vector<int> W,C;	//第i件物品的worth和cost	i从1开始 边界控制
vector<vector<int>> max_value(N+1,vector<int>(V+1,0));	//初始化为0 边界控制
int zero_one_pack()
{
    for(int i=1;i<=N;i++)
        for(int v=0;v<=V;v++)
            if(v<C[i])max_value[i][v]=max_value[i-1][v];	//无法放入
    		else max_value[i][v]=max(max_value[i-1][v],max_value[i-1][v-C[i]]+W[i]);
    return max_value[N][V];
}
```

### 空间优化

>  二维变为一维

```c++
int N,V;	//物品数和总容量
vector<int> W,C;	//第i件物品的worth和cost	i从1开始 边界控制
vector<int> max_value(V+1,0);	//初始化为0 边界控制
int zero_one_pack()
{
    for(int i=1;i<=N;i++)
        for(int v=V;v>=C[i];v--)	//逆序 保证取上一个的是未更新的
    		max_value[v]=max(max_value[v],max_value[v-C[i]]+W[i]);
    return max_value[V];
}
```

### 完全利用情况

> 假如问题改为背包空间需要完全利用,只需要在初始化时除了max_value[0]为0,其他为负无穷即可
>
> 因为初始化的数组事实上就是没有任何东西放入的合法状态,显然只有0空间时候是合法的
>
> 该思路其他背包问题也适用

```c++
int N,V;	//物品数和总容量
vector<int> W,C;	//第i件物品的worth和cost	i从1开始 边界控制
vector<int> max_value(V+1,INT_MIN);	//初始化为负无穷 边界控制
int zero_one_pack()
{
    max_value[0]=0;	//初始化0空间
    for(int i=1;i<=N;i++)
        for(int v=V;v>=C[i];v--)	//逆序 保证取上一个的是未更新的
    		max_value[v]=max(max_value[v],max_value[v-C[i]]+W[i]);
    return max_value[V];
}
```

## 完全背包

### 朴素

> 有$N$种物品和一个容量为$V$的背包,每种物品有无限件可用,放入第$i$件物品耗费的费用是$C_i$,得到的价值是$W_i$.求解将哪些物品装入背包可以使价值总和最大.

> 根据01背包转移方程进行思考,那么转移方程为
>
> $max\_value_{i,v}=max(max\_value_{i,v},max\_value_{i-1,v-kC_i}+kW_i)$
>
> 其中 $0\le kC_i \le v$
>
> 此处的第一种选择为考虑当前的上一个数量的选择,当$k=0$时,第二种选择即不选

```c++
int N,V;	//物品数和总容量
vector<int> W,C;	//第i件物品的worth和cost	i从1开始 边界控制
vector<vector<int>> max_value(N+1,vector<int>(V+1,0));	//初始化为0 边界控制
int complete_pack()
{
    for(int i=1;i<=N;i++)
        for(int v=0,k=0;v<=V;v++,k=-1)
            while(++k*C[i]<=v)	//选k个这个物品
                max_value[i][v]=max(max_value[i][v],max_value[i-1][v-k*C[i]]+k*W[i]);
    return max_value[N][V];
}
```

### 时空优化

> 在考虑第$i$种物品时,包含考虑加选一件第$i$种物品

```c++
int N,V;	//物品数和总容量
vector<int> W,C;	//第i件物品的worth和cost	i从1开始 边界控制
vector<int> max_value(V+1,0);	//初始化为0 边界控制
int complete_pack()
{
    for(int i=1;i<=N;i++)
        for(int v=C[i];v<=V;v++)	//正序 包含考虑当前重复选择
    		max_value[v]=max(max_value[v],max_value[v-C[i]]+W[i]);
    return max_value[V];
}
```

## 多重背包

> 有$N$种物品和一个容量为$V$的背包,每种物品有$M_i$个可用,放入第$i$件物品耗费的费用是$C_i$,得到的价值是$W_i$.求解将哪些物品装入背包可以使价值总和最大.

> 考虑第$i$种物品时
>
> 如果$M_iC_i\ge V$那么该种物品就可以当做是完全背包中的其中一个物品处理
>
> 否则,则可以把该种物品按照$1,2,2^2...2^{k-1},M_i-2^k+1$作为系数分成多个物品,那么这些物品就可以当做01背包来进行处理了

```c++
int N,V;	//物品数和总容量
vector<int> W,C,M;	//第i件物品的worth和cost和个数	i从1开始 边界控制
vector<int> max_value(V+1,0);	//初始化为0 边界控制
int multiple_pack()
{
    for(int i=1;i<=N;i++)
        if(M[i]*C[i]>=V)
            for(int v=C[i];v<=V;v++)	//完全背包
    			max_value[v]=max(max_value[v],max_value[v-C[i]]+W[i]);
    	else{
            int k=1;
            while(k<M[i]){
                for(int v=V;v>=k*C[i];v--)	//01背包
    				max_value[v]=max(max_value[v],max_value[v-k*C[i]]+k*W[i]);//乘上系数k
                M[i]-=k;
                k<<=1;
            }
             for(int v=V;v>=M[i]*C[i];v--)
    			max_value[v]=max(max_value[v],max_value[v-M[i]*C[i]]+M[i]*W[i]);
            	//最后进行一次系数为余下的M[i]为系数01背包
        }
    return max_value[V];
}
```

## 最长上升子序列

> 给定一个序列,求数值严格单调递增的子序列长度最长是多少

### 暴力

>$lis_i$表示以$a_i$结尾的最大上升序列长度
>
>转移方程为:$lis_i=max(lis_i,lis_j+1)\;(j∈(0,i-1))\;(a_i>a_j)$
>
>时间复杂度$O(n^2)$

```c++
int n;	//序列长度
vector<int> lis(n,0);
int LIS(vector<int> a)
{
    int ans=1;	//最小的
    for(int i=0;i<n;i++)
    {
        lis[i]=1;		//默认为1 即找不到前面比他小的数字
        for(int j=0;j<i;j++)
            if(a[i]>a[j])lis[i]=max(lis[i],lis[j]+1);	//进行转移 是否从j后面接上i更长
        ans=max(ans,lis[i]);	//此处要进行更新 无法知道哪个为结尾更长
    }
    return ans;
}
```

### 二分

> $lis_i$表示长度为i的上升序列集合中的最小的最后一位数
>
> 转移方程为:$lis_i=min(a_j)\;\;(a_j>lis_{i-1},j∈(pos_{lis_{i-1}}+1,n))$
>
> 由于j需要以某个起点往后找,因此可以从头到尾遍历$a_j$来更新最长的长度
>
> $lis$数组显然是单调递增的,因为序列长的在其后面新增的数字一定比前面的短序列数字要大
>
> 因此可以用二分法来找到第一个$lis_i$比$a_j$要小的
>
> 时间复杂度$O(nlogn)$

```c++
int n;	//序列长度
vector<int> lis(n,0);
int LIS(vector<int> a)
{
    lis[0]=INT_MIN;	//初始化0位
    int len=0;	//当前最长的序列长度
    for(int i=0;i<n;i++){
        if(a[i]>lis[len])lis[++len]=a[i];	//如果当前数比最长序列最后一位要长 即可以更长并且最后一位是当前数
        else{	//当前数无法更新最大长度
            int l=1,r=len;
            while(l<r){
                int mid=(l+r)>>1;
                if(lis[mid]>=a[i])r=mid;	//找到第一个大于或等于当前数的最后长度并用当前数去更新该长度的最后一位的最小长度
                else l=mid+1;
            }
            lis[r]=a[i];
        }
    }
    return len;
}
```

### Dilworth定理

> + 对于一个偏序集，最少链划分等于最长反链长度。
> + 对偶定理：对于一个偏序集，其最少反链划分数等于其最长链的长度。
>
> 那么可以知道划分成**最少**的**最长上升子序列**的**个数**就等于这个数列的**最长下降子序列**的**长度**

## 最长公共子序列

> 求给定两个序列s1,s2的最长公共子序列,并求得路径
>
> $lcs_{i,j}$表示s1中1..i且在s2中1..j的最长公共子序列长度
>
> 转移方程为:	lcs_{i,j}=lcs_{i-1,j-1}+1\; (s1_i=s2_j)=max(lcs_{$i$-1,j},lcs_{i,j-1})\; (s1_i\ne s2_j)
>
> 时间复杂度$O(n^2)$

```c++
int n,m;	//s1和s2的长度
vector<vector<int>> path(n+1,vector<int>(m+1,0));	//储存路径信息
vector<vector<int>> lcs(n+1,vector<int>(m+1,0));	//dp数组
int LCS(string s1,string s2)
{
    s1.insert(0,1,'#');
    s2.insert(0,1,'#');	//为了方便 下标从1开始
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
            if(s1[i]==s2[j]){	//当前两个相等
                lcs[i][j]=lcs[i-1][j-1]+1;	//直接从左上角转移
                path[i][j]=1;	//表示从左上角转移过来的
            }
    		else if(lcs[i-1][j]>lcs[i][j-1]){
                lcs[i][j]=lcs[i-1][j];
                path[i][j]=2;	//表示从上方转移
            }
    		else {
                lcs[i][j]=lcs[i][j-1];
                path[i][j]=3;	//表示从左方转移
            }
    return lcs[n][m];
}
void get_LCS(int i,int j,string &s1,string &ans)	//i j初始传入n m且ans为空字符串
{
    if(!i||!j)return;
    if(path[i][j]==1){	//表示相等
        get_LCS(i-1,j-1,s1,ans);
        ans.push_back(s1[i]);		//需要注意s1需要下标从1开始
    }
    else if(path[i][j]==2)get_LCS(i-1,j,s1,ans);
    else get_LCS(i,j-1,s1,ans);
}
```

## 最长公共上升子序列

### 朴素

> $lcis_{i,j}$表示以a中1..i且b中1..j的最长公共上升子序列的长度
>
> 转移方程为: $lcis_{i,j}=lcis_{i-1,j}\;(a_i\ne b_j)$	最后会枚举n与1..j的最大值,因此不需要对j进行判断
>
> $lcis_{i,j}=max(lcis_{i-1,k})+1\;(a_i=b_j)\;k∈(1,j-1)$
>
> 最后进行枚举: $lcis=max(lcis_{n,j})$
>
> 时间复杂度$O(n^3)$

```c++
int n,m;	//a,b序列的长度
vector<vector<int>> lcis(n+1,vector<int>(m+1,0));
int LCIS(vector<int> a,vector<int> b)		//要求序列下标从1开始
{
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            lcis[i][j]=lcis[i-1][j];	//不相等转移
            if(a[i]==b[j])	//最后相等
            {
                int maxv=1;
                for(int k=1;k<j;k++)	//枚举b序列中第几个可以作为上一个数
                    if(b[k]<a[i])maxv=max(maxv,lcis[i-1][k]+1);	//且求最大长度
                lcis[i][j]=max(lcis[i][j],maxv);
            }
        }
    int ans=0;
    for(int i=1;i<=m;i++)ans=max(ans,lcis[n][i]);	//因为最后不一定是以b[m]结尾的
    return ans;
}
```

### 小优化

> 把maxv提了出来
>
> 时间复杂度$O(n^2)$

```c++
int n,m;	//a,b序列的长度
vector<vector<int>> lcis(n+1,vector<int>(m+1,0));
int LCIS(vector<int> a,vector<int> b)		//要求序列下标从1开始
{
    for(int i=1;i<=n;i++)
    {
        int maxv=1;
        for(int j=1;j<=m;j++)
        {
            lcis[i][j]=lcis[i-1][j];
            if(a[i]==b[j])lcis[i][j]=max(lcis[i][j],maxv);
            if(a[i]>b[j])maxv=max(maxv,lcis[i-1][j]+1);
        }
    }
    int ans=0;
    for(int i=1;i<=m;i++)ans=max(ans,lcis[n][i]);
    return ans;
}
```

## 打家劫舍(不相邻子序列求最大和)

>给定n间房屋内藏有的现金数,且不能盗取相邻的房屋,问最多能偷窃到的最高金额
>
>定义$dp_i$为第$i$间及之前可以盗取的最大价值数
>
>转移方程为:$dp_i=max(dp_{i-1},dp_{i-2}+money_i)$
>
>即表示偷这间和上上间和不偷这间哪个比较多
>
>其中$dp_{i-1}$不一定表示第$i-1$间被偷,但是若没有被偷,此时$dp_{i-2}=dp_{i-1}$

```c++
int rob(vector<int> money)
{
    int n=money.size();
    if(!n)return 0;
    if(n==1)return money[0];		//特判n<2的
    vector<int> dp(n);
    dp[0]=money[0];
    dp[1]=max(dp[0],money[1]);
    for(int i=2;i<n;i++)
        dp[i]=max(dp[i-1],dp[i-2]+money[i]);	//偷还是不偷
    return dp[n-1];
}
```

> 假如房屋是首尾相接的话,那么可以分成1 ~ n-1和2 ~ n两段数组来跑两次上面的函数

> 如果需要把数量的状态也考虑的话只需要多加一维数量j 转移方程为$dp_{i,j}=max(dp_{i-1,j},dp_{i-2,j-1}+money_i)$

## 记忆化搜索

> 记忆化搜索本质是搜索的形式,动态规划的思想
>
> 一般来说记忆化搜索所需的记忆空间较大,优势在于有些逻辑不太好递推转移方程时候,搜索形式降低了难度
>
> 简单来说 **只要不爆栈 我就暴力搜**

```c++
const int n;
vector<int> mem(n,-1);	//记忆 类似于dp数组 使用一个标记值表示未曾搜索过
int ms(int pos)	//一般用一个答案类型作为函数返回值
{
    if(mem[pos]!=-1)return mem[pos];	//子问题 已经搜索过 直接返回答案
    mem[pos]=0;	//初始值 求最小子问题答案初始化为无穷大
    for(int i=0;i<max_pos;i++)
    {
        //搜索过程
        if(ok(pos,i))	//如果合法的下一步
        {
            //在搜索中找到最优解并赋值给记忆数组
            mem[pos]=max(mem[pos],ms(i));
        }
    }
    return mem[pos];
}
```

## 区间dp

题目的状态大多由`dp[l][r]`构成的 而且大区间可以划分成小区间来处理

### 最小分割花费

这类问题需要枚举分割点k  得到最小的分割花费

复杂度为$O(n^3)$

```cpp
int n;
vector<vector<int>> dp(n,vector<int>(n,0));
for(int len=1;len<=n;len++)		//先枚举小的区间
    for(int i=0;i<n;i++){	//枚举区间起点
   		int j=i+len-1;	//根据起点和长度得出终点
        if(j>=n)break;
        dp[i][j]=INT_MAX;
        for(int k=i;k<j;k++)
            dp[i][j]=min(dp[i][j],dp[i][k]+dp[k+1][j]+cost[i][j]);
    }
```

# 数论

## 筛法求素数

>  使用倍数筛法进行排除非素数,空间换时间

### 素数定理

> n以内的质数个数约为$\dfrac{x}{lnx}$

### Eratosthenes筛法

>  时间复杂度为$O(nloglogn)$
>
>  不必从i的2倍开始筛,因为2已经筛了,同理3倍也是,......,因此从i*i开始筛

```c++
const int n;
vector<int> x(n+1,0);	//存的是合数i的最大质因数 质数为0
void get_primes(int n){
    for(int i=2;i*i<=n;i++)	//从2开始到根号n
        if(!x[i])for(int j=i*i;j<=n;j+=i)x[j]=i;	//是质数把其后面的把他倍数都设置为非质数
}
```

### Euler筛法

> 时间复杂度为$O(n)$
>
> 保证每个合数只被他最小质因数筛过一次

```c++
const int n;
vector<int> x(n+1,0); //n的最小质因数	
vector<int> prime;	//质数表
void get_primes(int n){
    prime.clear();
    for(int i=2;i<=n;i++){
        if(!x[i]){
            x[i]=i;
            prime.push_back(i);		//质数表里插入质数i
        }
        for(int j=0;i*prime[j]<=n;j++){
            x[i*prime[j]]=prime[j];		//让i和质数相乘 之前必定没有筛过
            if(i%prime[j]==0)break;		//i的最小质因子是prime[j]
        }
    }
}
```

## 质因数分解

> 每一个合数都可以写成多个质数相乘的形式

### 直接分解法

```c++
map<int,int> factors;	//存放分解的质因数 和 他的次方
void factorize(int n) {
    factors.clear();
	for (int i = 2; i*i <= n; i++) {
		while (!(n%i)) {	//当还能被当前质数整除时
			factors[i]++;
			n /= i;
		}
		if (n == 1)return;	//1不是质数
	}
	if (n > 1)factors[n]++;	//最后剩下他本身
}
```

### 预处理最小质因数法

> $O(log\,n)$   其中$O(n)$是欧拉筛预处理的

```c++
vector<int> x;	//预处理最小质因数数组
map<int,int> factors;	//存放分解的质因数
//get_primes(n);	//获得最小质因数数组	欧拉筛
void factorize(int n){
    factors.clear();
    while(n>1){		//不断除最小质因数
        factors[x[n]]++;
        n/=x[n];
    }
}
```

### 阶乘的质因数

```c++
int get_factorial_power_k_of_p(int n,int prime)	//n!的质因数p p^k 返回k
{
    int k=0;
    while(n)k+=n/=prime;	//1*2*...*n 中含有n/p个p
    return k;
}
```

## 因数个数

> 一个数的质因数分解为
>
> $$a_1^{p_1}*a_2^{p_2}*a_3^{p_3}*...*a_n^{p_n}$$
>
> 那么这个数的因数个数为
>
> $$(p_1+1)*(p_2+1)*...*(p_n+1)$$

```cpp
map<int,int> factors;	//存放分解的质因数 和 他的次方
//factorize(n);		//进行预处理
int cnt_factors(int n)
{
    int cnt=1;
    for(auto i:factors)
    	cnt*=i.second+1;
    return cnt;
}
```

### 暴力计算

```cpp
int cnt_factors(int n)
{
    int m=sqrt(n);	//上限
    int cnt=0;
    for(int i=1;i<=m;i++)	//去找所有因子
        if(n%i==0){
            cnt++;	//发现一个因子
            if(n/i!=i)cnt++;	//只要不是平方数 那么肯定有另一个因子
        }
    return cnt;
}
```

## 最大公约数

> **欧几里得**求两个数的最大公约数
>
> 递归的层数最多是$gcd(Fib(n),Fib(n-1))$

### 递归版本

```c++
int gcd(int a,int b)
{
    if(a<b)return gcd(b,a); //保证a<b
    if(!b)return a; //除数为0结束
    return gcd(b,a%b);	//辗转相除
}
```

```c++
int gcd(int a,int b){return b?gcd(b,a%b):a;}
```

### 非递归版本

```C++
int gcd(int a,int b)
{
    if(a<b)swap(a,b);	//保证a<b
    while(b)	//当除数不为0
    {
        int t=b;
        b=a%b;
        a=b;		//a=b,b=a%b
    }
    return a;
}
```

## 最小公倍数

> 求两个数的最小公倍数

```C++
#define lcm(a,b) (a/gcd(a,b)*b)	//防止溢出
```

## 拓展欧几里得

<h4 name="exgcd">拓展欧几里得</h4>
> 用来解决:
> $$
> gcd(a,b)=ax+by
> $$
> 其中系数x和y的求解  此处ab均大于0  显然正负可以由系数决定

```c++
int exgcd(int a,int b,int &x,int &y) //x y 用来储存结果	返回值为gcd(a,b)
{
    if(!b){
        x=1,y=0;	//递归终点x=1,y=0
        return a;
    }
    int d=exgcd(b,a%b,y,x);	//进入递归时候会交换x,y位置
    y-=(a/b)*x;		//公式
    return d;
}
```

> 对于$ax+by=d$,当且仅当$d=k*gcd(a,b)$时候有解  解为（kx，ky）
>
> 设该方程的一个解为$(x_0,y_0)$,那么$(x_0-k*\dfrac{b}{gcd(a,b)},y_0+k*\dfrac{a}{gcd(a,b)})$也为该方程的一个解
>
> k为任意整数

## 快速幂

> $$
> A^B\%mod
> $$
>
> B可以分解成:
> $$
> 2^{a_12^0+a_22^1+a_32^3+...+a_n2^n}\\
> {a_n}=1 \ or \ 0
> $$

### 带模快速乘

> int * int 没有必要用 会减慢速度

```C++
int quick_mul(int a,int b,int mod)
{
    long long ans=0;	//防止溢出
    while(b){			//当b还有位数的时候
        if(b&1)ans=(ans+a)%mod;		//二进制最后一位判断1 or 0
        a=(a<<1)%mod;		//a=a+a
        b>>=1;		//舍弃最后一位
    }
    return (int)ans;
}
```

### 带模快速幂

```C++
int quick_pow(int a,int b,int mod)
{
    long long ans=1;
    while(b){
        if(b&1)ans=quick_mul((int)ans,a,mod);
        a=quick_mul(a,a,mod);
        b>>=1;
    }
    return (int)ans;
}
```

## 欧拉函数

> 欧拉函数,一般记作$\phi(n)$,表示小于等于n的数中与n互质的数的个数

> 如果 $n=p_1^{a_1}*p_2^{a_2}*...*p_m^{a_m}$ ,即$p_i$是n的质因数
>
> 则有$\phi(n)=n*(1-\dfrac{1}{p_1})*...*(1-\dfrac{1}{p_m})=n*(\dfrac{p_1-1}{p_1})*...*(\dfrac{p_m-1}{p_m})$

### 定义式实现

```c++
int get_euler(int n)	//返回n的欧拉函数
{
    int ans=n;		//式子中的第一个n
    for(int i=2;i*i<=n;i++)
        if(!(n%i)){		//如果是质因数
            ans=ans/i*(i-1);	//进行一项计算
            while(!(n%i))n/=i;	//把该质因数的幂数消掉
        }
    if(n>1)ans=ans/n*(n-1);	//最后的质因数
    return ans;
}
```

### Eratosthenes筛法

>筛法$O(n*loglogn)$求出1-n的欧拉函数

```c++
int n;
vector<int> euler(n+1,0);
void get_eulers(int n)	//本质是利用定义来算
{
    for(int i=1;i<=n;i++)euler[i]=i;	//初始化
    for(int i=2;i<=n;i++)
        if(euler[i]==i)	//代表i是质数
            for(int j=i;j<=n;j+=i)
                euler[j]=euler[j]/i*(i-1);	//i是j的质因数 更新掉
}
```

> 欧拉函数的常用性质:
>
> 1. 如果$(n,m)=1$,则$\phi(n*m)=\phi(n)*\phi(m)$
>
> 2. 小于$n$的,且和$n$互质的数的数的和是$\dfrac{\phi(n)*n}{2}$
>
> 3. 如果$n$是质数,$\phi(n)=n-1$
>
> 4. $p$是质数,$n=p^k$,则有$\phi(n)=p^k-p^{k-1}$
>
> + **欧拉定理**:如果$n,a$互质,且均为正整数,则有$a^{\phi(n)}\equiv1(mod\;n)$
> + **费马小定理**:对于质数$p$,任意整数$a$.均有$a^{p-1}\equiv 1\;(mod\;p)$
> + **欧拉定理推论**:若正整数$n,a$互质,对于任意正整数b,有$a^b\equiv a^{b\;mod\;\phi(n)}\;(mod\;n)$

### Euler筛法

> 利用一些性质,可以在$O(n)$的时间复杂度求出1~n的所有欧拉函数

```c++
int n;
vector<int> primes,euler(n+1,0);	//质数表和欧拉函数表
vector<int> x(n+1,0);	//存的是最小的质因数
void get_eulers(int neu)
{
    euler[1]=1;	//特殊判断
    for(int i=2;i<=n;i++)
    {
        if(!x[i]){		//如果是质数
            primes.push_back(i);
            euler[i]=i-1;	//利用性质3 质数的欧拉函数为质数-1
        }
        for(int j=0;i*primes[j]<=n;j++){
            x[i*primes[j]]=primes[j];	//先把这个合数标记掉
            if(i%primes[j]==0){
                euler[i*primes[j]]=euler[i]*primes[j];	//i是当前质数的倍数 证明phi(i)已经包含i*primes[j]的全部质数 根据定义式只需要在最前面乘个primes[j]
                break;	//保证每个数只会被自己最小的质因数筛掉一次
            }
            euler[i*primes[j]]=euler[i]*euler[primes[j]];	//不能整除证明是互质 性质1
        }
    }
}
```

## 逆元

> 一个数x在模p的情况下的逆元

### 费马小定理快速幂求逆元

> 对于质数$p$,任意整数$a$.均有$a^{p-1}\equiv 1\;(mod\;p)$
>
> 易得$inv(a)*a\equiv a^{p-1}$, 即$inv(a)\equiv a^{p-2}$
>
> 限制大,mod必须为质数

```c++
int inv(int a,int mod){
    return quick_pow(a,mod-2,mod);
}
```

### 扩展欧几里得求逆元

> $a*inv(a)=1(mod\;p)$ 令$inv(a)=x$
>
> $ax=1+py$	p是未知的系数
>
> $ax-py=1$
>
> 显然上式可以用扩展欧几里得计算出系数x和y

```c++
int inv(int a,int mod){
    int inv_a,tmp;	//系数y在此处无关
    ex_gcd(a,mod,inv_a,tmp);
    return (inv_a%mod+mod)%mod;		//系数x有可能是负数
}
```

### 组合数

> 1. $C_n^m=\dfrac{n!}{m!(n-m)!}$
> 2. $C_n^m=C_n^{n-m}$
> 3. $C_n^m=C_{m-1}^m+C_{n-1}^{m-1}$

### 公式求解

> 利用公式1和公式2 **直接**求

```c++
unsigned long long C(int n,int m)
{
    if(m>n-m)return C(n,n-m);	//公式2
    long long ans=1;
    for(int i=1;m>0;m--,i++,n--)	//顺序不能错 因为整除的性质
        ans=ans*n/i;	//进行约分	分子从n开始倒数m个	分母从1开始到m
    return ans;
}
```

### 杨辉三角

> 公式3 **杨辉三角**求0-n层的全部组合数

```c++
const int n; //0-n层
vector<vector<int>> C(n+1,vector<int>(n+1,0));	//二维组合数
void getC(int n)
{
    for(int i=0;i<=n;i++)
        for(int j=0;j<=i;j++)
        	if(!j||i==j)C[i][j]=1;
    		else C[i][j]=C[i-1][j]+C[i-1][j-1];
}
```

### 分解质因数

> 求高精度组合数
>
> 需要用到筛质数函数、大数相乘函数和求阶乘的质因数个数函数

```c++
vector<int> prime;	//质数组
vector<int> getC(int n,int m)	//返回的是倒序的大整数
{
    get_primes(n);	//先获取n以内的质数
    vector<int> C={1};	//答案数组
    for(int i=0;i<(int)prime.size();i++)	//对于每个n以内的质数
    {
        int n_num=get_factorial_power_k_of_p(n,prime[i]);	//n!质因子中prime[i]的个数
        int m_num=get_factorial_power_k_of_p(m,prime[i]);	//m!质因子中prime[i]的个数
        int n_m_num=get_factorial_power_k_of_p(n-m,prime[i]);	//(n-m)!质因子中prime[i]的个数
        int factors=n_num-m_num-n_m_num;	//相减
        while(factors--)mul(C,prime[i]);	//大整数相乘 把质因子都乘进答案
    }
    return C;
}
```

### 逆元求带模组合数

> 求$C(n,m)\%p$的值,其中$p$是质数,那么原始为: $n!*m!^{p-2}*(n-m)!^{p-2}\%p$

```c++
int C(int n,int m,int p)	//p要为质数
{
    long long c=1;
    for(int i=2;i<=n;i++)c=(c*i)%p;
    for(int i=2;i<=m;i++)c=c*inv(i,p)%p;
    for(int i=2;i<=n-m;i++)c=c*inv(i,p)%p;
    return (int)c;
}
```

## 排列数

### 全排列组合

> **next_permutation**这个函数可以把一个序列进行全排列,其返回值是一个bool,如果还存在下一个排列方式就对序列进行更改并返回true,该函数还接受第二个参数传入一个自定义的compare函数

```c++
long long get_permutation(vector<int> arr,vector<vector<int>> &ans)	//返回的是排列数
{
    long long cnt=0;
    sort(arr.begin(),arr.end());	//先进行排序 保证从升序开始
    do{
        cnt++;
        ans.push_back(arr);
    }while(next_permutation(arr.begin(),arr.end()));	//全排列函数
    return cnt;
}
```

### 全错位排列

> 递推公式为:$D_n=(n-1)(D_{n-1}+D_{n-2})$

```c++
long long all_dislocation_arr(int n)
{
    long long d1=0,d2=1;
    if(n<=1)return d1;
    if(n==2)return d2;
    long long ans;
    for(int i=3;i<=n;i++){
        ans=(i-1)*(d2+d1);
        d1=d2;
        d2=ans;
    }
    return ans;
}
```

## 卡特兰数

>  1, 1, 2, 5, 14, 42, 132, 429, 1430....

> 卡特兰数相关公式
>
> 1. $H_n=\dfrac{C_{2n}^n}{n+1}=C_{2n}^n-C_{2n}^{n-1}$
> 2. $H_n=\dfrac{4n-2}{n+1}*H_{n-1}\;\;H_{0,1}=1$
> 3. $H_n=\sum_{i=1}^nH_{i-1}H_{n-i}\;(n\gt1)\;\;\;\;H_{0,1}=1$

### 公式求解

> 利用公式 1 **直接求**

```c++
unsigned long long catalan(int n)
{
    //此处的组合数可以用杨辉三角或者直接求得
    return C(2*n,n)/(n+1);
}
```

### 递推求解

> 利用公式2 

```c++
unsigned long long catalan(int n)
{
    unsigned long long cat=1;	//第0项
    for(int i=1;i<=n;i++)
        cat=cat*(4*i-2)/(i+1);
    return cat;
}
```

### 高精度卡特兰数

```c++
vector<int> catalan(int n)
{
    vector<int> C=getC(2*n,n);	//因为div的原型是A和r都是引用
    return div(C,n+1,n);	//公式1
}
```

## 星期x

> Kim Iarsen 公式
>
> $W=(d+2m+3(m+1)/5+y+y/4-y/100+y/400+1)\;mod\;7$
>
> 其中特别的,该公式把1月和2月看作是13月和14月
>
> 0-6 对应的是星期日-星期六

```c++
const string week_day[]={"Sun","Mon","Tue","Wed","Thu","Fri","Sat"};
string get_week(int y,int m,int d)
{
    if(y<0||m<1||m>12||d<1)return "wrong day";
    if(m==1||m==3||m==5||m==7||m==8||m==10||m==12){if(d>31)return "wrong day";}
    else if(m==2){
        if((y%4==0&&y%100!=0)||y%400==0){if(d>29)return "wrong day";}
        else if(d>28)return "wrong day";
    }
    else if(d>30)return "wrong day";
    m<3?m+=12,y--:0;
    return week_day[(d+2*m+3*(m+1)/5+y+y/4-y/100+y/400+1)%7];
}
```

# 图论

## 图的储存

### 邻接矩阵

```C++
const int n,m; 	//n个节点 m条边
vector<vector<int>> G(n,vector<bool>(n,0x3f3f3f3f));	//节点和节点的对应关系 无通路为无穷大
#define add_edge(a,b,w) (G[a][b]=w)
```

### 邻接表

> 对重复边会重复加入
>
> head其实会不断指向新加进来的边 而Next则是把更换的head边给保存下来 所谓链表的头插法

```C++
const int n,m;	//n个节点 m条边
vector<int> edge(m+5,0);	//存编号为i的边的终点节点  a-->b  存的是b的值
vector<int> Next(m+5,0);	//存编号为i的边的下一个兄弟节点编号	a-->b	a-->c	存的是下一条边的编号
vector<int> head(n+5,-1);	//存节点i的第一条边的编号 a-->b -1为无边
vector<int> weight(m+5,0);	//存编号为i的边的权重
int idx=0;		//编号索引i
void add_edge(int a,int b,int w)		//添加边
{
    edge[idx]=b;		//边i的终点是b
    weight[idx]=w;		//边i的权重是w
    Next[idx]=head[a];	//边i的下一个是之前同样为a节点的编号
    head[a]=idx++;		//a的编号是idx
}
void work()		//操作邻接表
{
    for(int i=1;i<=n;i++)		//对于每个节点去遍历邻接边
        for(int j=head[i];~j;j=Next[j])		//从节点i的第一条边开始
        {
            //操作
        }
}
```

### 前向星

> 点多的情况有优势

```c++
int n,m;	//点数 边数
vector<int> head(n,-1);
typedef struct node{
    int from,to,weight;	//起点终点和权重
    bool operator < (const node b) const	//重载小于号
    {
        if(from==b.from){
            if(to==b.to)
                return weight<b.weight;
            return to<b.to;
        }
        return from<b.from;
    }
}node;
vector<node> edge(n);
void add_edge(int i,int a, int b, int w)	//添加边
{
	edge[i].from = a;
	edge[i].to = b;
	edge[i].weight = w;
}
void graph_update()
{
    sort(edge.begin(),edge.begin()+m);	//根据起点排序
    head[edge[0].from]=0;	//第一个起点
    for(int i=1;i<m;i++)
        if(edge[i].from!=edge[i-1].from)head[edge[i].from]=i;	//下一个点的第一条边
}
void work()		//操作图
{
    for(int i=0;i<n;i++)		//对于每个节点去遍历边
        for(int j=head[i];edge[j].from==i&&j<m;j++)		//从节点i的第一条边开始
        {
            //操作
        }
}
```

## 拓扑排序

> 对于一个有向无环图G进行拓扑排序,将G所有顶点排成一个线性序列,使得图中任意一对顶点a和b,若边<a,b>∈edge,a一定出现在b之前

> 通过bfs来进行排序

```c++
const int n,m;	//n个节点 m条边
vector<int> topo;	//结果数组
vector<int> edge(m+5,0);	//边
vector<int> Next(m+5,0);	//下一条兄弟边
vector<int> head(n+5,-1);	//节点的链表
vector<int> in(n+5,0);		//节点的入度
//假设已经建好邻接表
bool topo_sort(){
    //priority_queue<int,vector<int>,greater<int>> q;
    //优先队列可以使无冲突节点按顺序输出
    queue<int> q;
    for(int i=1;i<=n;i++)
        if(!in[i])q.push(i);	//寻找入度为0的节点 加入队列中
    while(q.size()){
        int a=q.front();	//找到第一个入度为0的		//优先队列q.top()
        q.pop();
        //在此处可以进行操作
        topo.push_back(a);
        for(int i=head[a];~i;i=Next[i]){	//遍历和a连接的所有点
            int b=edge[i];				//遍历到的b点
            if(--in[b]==0)q.push(b);	//b剔除掉边ab 入度-1 假如入度减到0 加入队列
        }
    }
    return topo.size()==n;		//如果所有点都入队了,说明存在拓扑排序,否则不存在
}
```

## 深度优先搜索

### 一般形式

```c++
void dfs(int status,int pos)
{
    if(status>=max_status||pos>=max_pos)	//深搜状态到底
    {
        //深搜到底的操作
        return;
    }
    if(invalid(status,pos)||vis[pos])
    {
        //状态不合法
        return;
    }
    if(!check(status,pos))
    {
        //其他状态剪枝
        return;
    }
    for(int i=pos;i<max_pos;i++)	//进行搜索节点的扩展
    {
        if(ok(status,pos))	//扩展方式合法
        {
            vis[i]=true;	//标记为是否走过	可选
            //遍历这个节点会有怎样的操作
            dfs(status+1,i+1);	//进行下一个状态的搜索
            vis[i]=false;	//还原标记	有这一步操作就是回溯
        }
    }
}
```

### 图的可达性判断

```c++
const int n,m,begin_x,begin_y;
vector<vector<int>> G(n,vector<int>(m));	//图 1为可走 0为不可走	2为终点
vector<vector<bool>> vis(n,vector<bool>(m,false));	//访问数组
int dir[4][2]={{-1,0},{1,0},{0,-1},{0,1}};	//四个方向上下左右
vis[begin_x][begin_y]=true;	//初始时起点标记为走过
bool graph_dfs(int x,int y)
{
    if(x<0||y<0||x>=n||y>=m)return false;
    if(!G[x][y])return false;	//不能走
    if(G[x][y]==2)return true;	//找到目标
    for(int i=0;i<4;i++)
    {
        int x_=x+dir[i][0],y_=y+dir[i][1];
        if(x_<0||y_<0||x_>=n||y_>=m)continue;	//越界
        if(vis[x_][y_])continue;
        vis[x_][y_]=true;	//标记走过
        if(graph_dfs(x_,y_))return true;
        vis[x_][y_]=false;	//标记还原 回溯
    }
    return false;	//全部情况均不可行
}
```

## 广度优先搜索

### 一般形式

```c++
void bfs()
{
    queue<int> q;	//广搜的队列
    for(auto i:start)	//起点加入队列
    {
        q.push(i);		//加入队列
        vis[i]=true;	//起点表示已经访问	可选
    }
    while(!q.empty())	//当队列不空的时候  //可以写while(q.size())
    {
        int len=q.size();	//获取当前层数的节点数	可选
        while(len--)
        {
            auto tmp=q.front();		//获取下一个要处理的节点
            q.pop();
            //进行一些操作
            for(auto i:Next)	//遍历下一层
            {
                if(ok(i)){		//节点可行的话就加入队尾
                    vis[i]=true;
                    q.push(i);
                }
            }
        }
    }
}
```

### 图的最短可达路径

```c++
const int n,m,begin_x,begin_y;
vector<vector<int>> G(n,vector<int>(m));	//图 1为可走 0为不可走	2为终点
vector<vector<bool>> vis(n,vector<bool>(m,false));	//访问数组
int dir[4][2]={{-1,0},{1,0},{0,-1},{0,1}};	//四个方向上下左右
int shortest_path()		//如果不可达 返回-1  否则返回最短距离
{
    queue<pair<int,int>> q;		//坐标对的队列
    q.push({begin_x,begin_y});
    vis[begin_x][begin_y]=true;
    int dis=-1;		//初始化距离
    while(q.size())
    {
        dis++;
        int len=q.size();
        while(len--)
        {
            int x=q.front().first,y=q.front().second;	//获取当前的坐标
            q.pop();
            if(G[x][y]==2)return dis;	//找到终点
            for(int i=0;i<4;i++)
            {
            	int x_=x+dir[i][0],y_=y+dir[i][1];	//获取下一个可能的坐标
                if(x_<0||y_<0||x_>=n||y_>=m)continue;	//越界
                if(vis[x_][y_]||!G[x_][y_])continue;	//走过或者不能走
                vis[x_][y_]=true;	//标记为走过
                q.push({x_,y_});	//下一个坐标加入队尾
        	}
        }
    }
    return -1;  //在遍历过程中没有找到可行的道路
}
```

## 树的直径

### BFS/DFS

> 两次搜索
>
> 第一次从任意一个点a出发进行搜索 找到离他最远的点x
>
> 然后从x出发 进行第二次搜索 找到离他最远的点y
>
> xy的路径就是树的直径

> 证明  从一个点出发必定能走到最远的为x或者y点
>
> 假设其走到最远点为z点 那么显然有xz或yz的路径长度大于xy
>
> 这样就与假设xy为直径不符合了

```cpp
int n;	//点的数量
vector<vector<int>> G;	//邻接表
vector<bool> vis;
int d=0,node=-1;	//直径长度和xy点
void dfs(int x,int len)
{
    if(vis[x])return;
    if(len>d){
        d=len;	//更新直径  第二次有用
        node=x;	//更新最远的点 第一次有用
    }
    vis[x]=true;
    for(int i=0;i<G[x].size();i++)
        dfs(G[x][i],len+1);	//如果该树有权值的话  len+1 改为len+两点的权值
}
int getTreeD(){
    dfs(a,0);	//从任意点a出发
	dfs(node,0);	//从最远点出发
	return d;
}
```

### 树形dp

> + 先取任意点为根 通常取0号点
> + 定义`dp[x]`为 以x为根节点的子树 (子树是相对上一步选取的根节点的) 的高度
> + 显然高度可以递归来获取
> + 在更新高度的同时 (记住 是同时)  可能某个节点x的`dp[x]`会经过另一条分支递归回来 从而比原来的`dp[x]`要大 那么这时候显然原来的高度值为次大值 现在的为最大值  那么此时定义`d[x]`为经过x节点的最长链长度 `d[x]=dp[x]'+dp[x]`
> + 最终的答案树的直径即为 `d=max{d[i]}(1<=i<=n)`

```cpp
vector<vector<int>> G,w;	//邻接表和权值
vector<bool> vis;	//假如把图建成有根有向 则不需要这个
vector<int> dp,d;
int ans=0;
void dfs(int x){
    if(vis[x])return;
    vis[x]=true;
    for(int i=0;i<G[x].size();i++){
        int &y=G[x][i];
        dfs(y);
        if(dp[y]+w[x][y]>dp[x]){	//y这条路径比原来的长
            d[x]=dp[x]+dp[y]+w[x][y];	//更新一下次大和最大形成的链
            dp[x]=dp[y]+w[x][y];	//记得先更新d再更新dp
            ans=max(ans,d[x]);
        }
    }
}
```

## 最短路径

### 朴素dijkstra

> 有向图中求一个点与其他点的最短距离,即单源最短路,且边权重只能为**正**
>
> 时间复杂度为$O(n^2)$,$n$为节点数,$m$为边数

```c++
int n;	//节点数
vector<vector<int>> G;	//有向图中邻接矩阵储存点与点的权值	无边时为无穷大
vector<int> path(n);	//保存最短路中点的前导节点
vector<int> dijkstra()	//0号节点作为起点
{
    vector<int> dis(n,0x3f3f3f3f);	//初始化为无穷大
    vector<bool> visited(n,false);	//初始化访问标记
 	dis[0]=0;	//0号点自身
    for(int i=0;i<n-1;i++){	//处理全部点 最后的点不用处理
        int node=-1;	//哪个点当前与起点最近
        for(int j=0;j<n;j++)
            if(!visited[j]&&(node==-1||dis[node]>dis[j]))
                node=j;	//寻找未处理过的点中距离最短的点
        for(int j=0;j<n;j++)	//用node的点去更新点
            //dis[j]=min(dis[j],dis[node]+G[node][j]);
            if(dis[node]+G[node][j]<dis[j]){
                dis[j]=dis[node]+G[node][j];
                path[j]=node;
            }
        visited[node]=true;	//标记为已访问
    }
    return dis;
}
void get_path(int x,vector<int> &Path)
{
    if(!x){
        Path.push_back(0);
        return;
    }
    if(path[x]!=-1)get_path(path[x],Path);
    Path.push_back(x);
}
```

### 堆优化dijkstra

>
>使用邻接表存图
>
>时间复杂度$O(mlogm)$,$m$为边数

```c++
const int n,m;	//n个节点 m条边
vector<int> edge(m+5,0);	//边
vector<int> weight(m+5,0);	//权重
vector<int> Next(m+5,0);	//下一条兄弟边
vector<int> head(n+5,-1);	//节点的链表
vector<int> dijkstra()	//0号节点作为起点
{
    vector<int> dis(n,0x3f3f3f3f);	//初始化为无穷大
    vector<bool> visited(n,false);	//初始化访问标记
    dis[0]=0;
    priority_queue<pair<int,int>,vector<pair<int,int>>,greater<pair<int,int>>> heap;
    heap.push({0,0});	//first是距离 second是节点编号
    while(heap.size())
    {
        int node=heap.top().second;	//取出堆顶的点
        heap.pop();
        if(visited[node])continue;	//如果已经更新过就不用了
        for(int i=head[node];~i;i=Next[i]){
            int now_node=edge[i],w=weight[i];
            if(dis[now_node]>dis[node]+w){	//可以更新
                dis[now_node]=dis[node]+w;
                heap.push({dis[now_node],now_node});
            }
        }
        visited[node]=true;
    }
    return dis;
}
```

### Bellman-Ford

>复杂度比Dijkstra高,但可以处理**带负权图**和发现**负环**
>
>判断负环时无需初始化dis

```c++
int n,m;	//节点数和边数
vector<node> edge(n);	//前向星储存结构
vector<int> dis(n,0x3f3f3f3f);	//起点到任意点的最短距离
bool bellman_ford()	//如果存在负环返回false
{
    bool flag;
    dis[0]=0;		//将第一个点置为0
    for(int i=0;i<=n;i++){		//迭代n+1次
    	flag=false;	//如果没有再能松弛则停止
        for(int j=0;j<m;j++){	//遍历每一条边
            if(dis[edge[j].from]>dis[edge[j].to]+edge[j].weight){
                dis[edge[j].from]=dis[edge[j].to]+edge[j].weight;
                flag=true;
            }
        }
        if(!flag)return true;	//已经无法再松弛
    }
    return false;	//如果还能更新 那么肯定有负环
}
```

### SPFA

> 队列优化的Bellman-Ford

```c++
const int n,m;	//n个节点 m条边
vector<int> edge(m+5,0);	//邻接表
vector<int> Next(m+5,0);
vector<int> head(n+5,-1);
vector<int> weight(m+5,0);
vector<int> dis(n+1,0x3f3f3f3f);	//起点到每个点的距离
bool SPFA()		//有负环返回false
{
    vector<bool> existed(n+1,false);	//记录是否已经在队列里
    dis[0]=0;	//初始化第一个
    queue<int> q;
    q.push(0);	//把起点放入
    int cnt=0;	//迭代次数
    while(q.size()){
        cnt++;
        if(cnt>n)return false;	//出现负环
        int t=q.size();
        while(t--){		//该层迭代
            int node=q.front();
            q.pop();
            existed[node]=false;	//出队还原标记
            for(int i=head[node];~i;i=Next[i])	//遍历与node相邻的节点
            {
                int now_node=edge[i];	//遍历到的节点
                if(dis[now_node]>dis[node]+weight[i])	//可松弛
                {
                    dis[now_node]=dis[node]+weight[i];
					if(!existed[now_node]){			//如果队列中已经存在该点 则不需要重复插入
                        q.push(now_node);
                        existed[now_node]=true;
                    }
                }
            }
        }
    }
    return true;
}
```

### Floyd

> 利用动态规划思想 求**任意**两点之间的**最短距离**和**最短路径**
>
> 不能处理负环 时间复杂度为$O(n^3)$

> **证明dp正确性**
>
> 设i,j最短路两端点之间**最大**编号节点为k,设i,k之间最大编号节点为k1,设k,j之间最大节点为k2
>
> 那么显然有k>k1且k>k2 而k1和k2在k之前肯定被dp过了
>
> 根据数学归纳 初始边界为当i,j之间是原子距离 无中间点时候

```c++
int n;	//节点数
vector<vector<int>> dis;	//初始时为邻接矩阵	无通路为无穷大	自身距离为0
vector<vector<int>> path(n,vector<int>(n,-1));	//路径	-1表示有直接的连边
void floyd()
{
    for(int k=0;k<n;k++)	//遍历中间点
        for(int i=0;i<n;i++)
            for(int j=0;j<n;j++)
                if(dis[i][j]>dis[i][k]+dis[k][j]){	//可松弛
                    dis[i][j]=dis[i][k]+dis[k][j];	//更新距离
                    path[i][j]=k;		//记录中间点
                }
}
void get_path(int u,int v,vector<pair<int,int>> &Path)	//获得从u到v的最短路径
{
    if(path[u][v]==-1)Path.push_back({u,v});
    else {
        int mid=path[u][v];		//对于中间节点去递归获取
        get_path(u,mid,Path);
        get_path(mid,v,Path);	
    }
}
//矩阵对称性优化+跳过无效路径
void floyd_2()
{
    for(int k=0;k<n;k++)
        for(int i=0;i<n;i++){
            int t=dis[i][k];
            if(t==inf)continue;
            for(int j=0;j<=i;j++){
                dis[i][j]=min(dis[i][j],t+dis[k][j]);
                dis[j][i]=dis[i][j];
            }
        }
}
```

### 多条最短路径

示例用dijkstra来存下最路径的条数

只需要在**松弛操作**的时候把相同长度的路径进行计数即可

```cpp
void dijk() {
	vector<int> dis(n, 0x3f3f3f3f);
	vector<int> vis(n, false);
	priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> heap;
	dis[c1] = 0;	//c1为起点
	cnt[c1] = 1;
	heap.push({ 0,c1 });
	while(heap.size()) {
		int node = heap.top().second;
		heap.pop();
		if (vis[node])continue;
		for (int i = 0; i < g[node].size(); i++) {
			int j = g[node][i].first, w = g[node][i].second;
			if (dis[j] > dis[node] + w) {
				dis[j] = dis[node] + w;
				cnt[j] = max(1, cnt[node]);	//因为当前只有node到j这条路 所以最多就是把node的路径数加上
				heap.push({ dis[j],j });
			}
			else if (dis[j] == dis[node] + w) {
				cnt[j] += cnt[node];	//除了node到j 还有其他到j的 计数加上
			}
		}
		vis[node] = true;
	}
}
```

## 最小生成树

> 带权连通图中总权值最小的生成树

### prim

> 每次找到与MST最近的节点纳入MST
>
> 复杂度$O(n^2+m)$

```c++
int n;	//节点数
vector<vector<int>> G;	//邻接矩阵
int prim()		//如果图不连通 返回无穷大 否则返回最小生成树的总权值
{
    vector<int> dis(n,0x3f3f3f3f);	//未加入MST的节点与MST的距离
    vector<bool> MST(n,false);	//每个节点是否在生成树当中
    int w=0;	//权值和
    for(int i=0;i<n;i++)
    {
        int node=-1;
        for(int j=0;j<n;j++)
            if(!MST[j]&&(node==-1||dis[node]>dis[j]))
                node=j;		//找到离生成树最小距离的一个点
        if(i&&dis[node]==0x3f3f3f3f)return 0x3f3f3f3f;	//不连通
        if(i)w+=dis[node];	//更新总权值
        MST[node]=true;	//加入节点到最小生成树
        for(int j=0;j<n;j++)
            dis[j]=min(dis[j],G[node][j]);	//更新距离 对于新加入的节点 更新未加入节点的边
    }
    return w;
}
```

### Kruskal

> 复杂度$O(mlogm)$
>
> 每次把权值最小的边纳入考虑,假如加入该边不会产生环,那么就是可行的

```C++
int n,m;	//节点数和边数
vector<int> f(n);	//并查集的祖先节点数组
struct edge{
    int from,to,weight;
    bool operator < (edge &u) const
    {
        return weight<u.weight;
    }
};
vector<edge> edges(m);
int Kruskal()	//如果无法生成树返回无穷大
{
    sort(edges.begin(),edges.begin()+m);		//按权重从小到大排序
    init();	//初始化并查集
    int ans=0;	//总权值
    int cnt=0;	//生成树有多少条边
    for(int i=0;i<m;i++)
    {
        int from=edges[i].from,to=edges[i].to,w=edges[i].weight;
        int fa=find(from),fb=find(to);	//分别找到祖先
        if(fa!=fb)	//如果不连通
        {
            Union(from,to);	//合并
            ans+=w;
            cnt++;
        }
    }
    if(cnt<n-1)return 0x3f3f3f3f;	//如果少于n-1条边 证明有不连通的
    return ans;
}
```

## 二分图

### 染色法判断二分图

> 选定任意未染色节点,染上一种颜色,然后遍历所有相邻节点染上另一种颜色,如果在期间发现相邻节点染上相同颜色,那么表示该图不是二分图

```c++
int n;	//节点数
int m;	//边数
vector<int> edge(m+5,0);	//存编号为i的边的终点节点 
vector<int> Next(m+5,0);	//存编号为i的边的下一个兄弟节点编号	
vector<int> head(n+5,-1);	//存节点i的第一条边的编号 
vector<int>	color(n,-1);	//-1表示未染色 0表示黑色 1表示白色
bool paint(int u,int c)	//深搜去染色
{
    color[u]=c;		//进行染色
    for(int i=head[u];~i;i=Next[i]){	//对相邻的节点进行染色
        int node=edge[i];
        if(color[node]==-1){	//如果未染色
            if(!paint(node,!c))return false;	//进行深搜染色 如果失败则返回
        }
        else if(color[node]==c)return false;	//同一个颜色 染色失败
    }
    return true;
}
bool is_bipartite_graph()
{
    for(int i=0;i<n;i++)
        if(color[i]==-1)		//对未染色节点进行检查
            if(!paint(i,0))return false;	//染色失败
    return true;
}
```

### 匈牙利算法求二分图最大匹配

```c++
int n1,n2;	//分别表示二分图里两个集合的点数
vector<int> head(n,-1);		//邻接表的三个数组
vector<int> edge(m,0);		//匈牙利算法只会用到第一个集合指向第二个集合的边
vector<int> Next(m,0);
vector<int> match(n2,-1);	//表示第二个集合的节点当前所对应着的第一个集合的节点 -1为没有匹配
vector<bool> visited(n2,false);	//表示是否在一次匹配中访问过节点 防止重边重复访问
bool find(int v)	//进行增广路径的寻找
{
    for(int i=head[v];~i;i=Next[i]){
        int node=edge[i];
        if(!visited[node]){
            visited[node]=true;
            if(match[node]==-1||find(match[node])){	//对应的点无匹配 或者可以使其他节点让位
                match[node]=v;	//定下这个匹配表示是可行的
                return true;
            }
        }
    }
    return false;
}
int hungary()	//求最大匹配数
{
    int ans=0;
    for(int i=0;i<n1;i++){
        visited=vector<bool>(n2,false);	//重置
        if(find(i))ans++;	//以此匹配成功
    }
    return ans;
}
```

# 字符串

## KMP

> $O(m+n)$的字符匹配,其中$m$是字符串的长度,$n$是匹配串的长度

### Next 数组预处理

> 性质: $(i+1)-Next[i]$ 即前缀的长度减去对应的Next为前缀循环节的大小

```c++
string pattern;	//模式子串
vector<int> Next(pattern.length(),0);	//Next数组
void Next_pre(string p,vector<int> &Next)
{
    for(int i=1,j=0;i<(int)p.length();i++)	//i是当前遍历到的后缀 j是前缀
    {
        while(j&&p[i]!=p[j])j=Next[j-1];	//跳转
        if(p[i]==p[j])j++;	//前后缀一样 前缀开始向后
        Next[i]=j;	
    }
}
```

### 匹配

```c++
string s;	//待匹配字符串
string pattern; //模式子串
int KMP_match(string s,string p,int begin)	//返回的是匹配成功的索引起点
{
    vector<int> Next(p.length(),0);
    Next_pre(p,Next);	//预处理出跳转数组
 	for(int i=begin,j=0;i<(int)s.length();i++)
    {
        while(j&&s[i]!=p[j])j=Next[j-1];	//当前匹配失败 跳转
        if(s[i]==p[j])j++;	//匹配成功 进行一下个
        if(j==(int)p.length())
        {
            //j=Next[j-1];	//有需要的话继续下一次匹配
            //匹配成功后的操作
            return i-(int)p.length()+1;	//返回在s中匹配到的下标
        }
    }
    return -1;	//匹配不成功
}
```

## Trie树

> 单词查找树,利用字符串的公共前缀来减少查询时间
>
> 复杂度为$O(n)$

> 利用数组来建树

```c++
const int N;	//树的最大节点数
const int M=26;	//树的子节点的数目 举例26为小写字母作为可能的子节点
vector<vector<int>> trie(N,vecotr<int>(M,0));	//trie树数组 存的是索引
vector<int> cnt[N];	//储存终点节点的个数
int idx=0;	//0是根节点且为空
void trie_insert(string s)
{
    int p=0;	//索引指针 从根节点开始
    for(int i=0;i<(int)s.length();i++)
    {
        int word=s[i]-'a';	//假设是只含有小写字母的例子
        if(!trie[p][word])trie[p][word]=++idx;	//如果当前没有这个节点就新创建一个
        p=trie[p][word];	//进入到这个节点
    }
    cnt[p]++;	//以这个节点结束的标记一下
}
int trie_query_times(string s)	//查询字符串出现的次数
{
    int p=0;	//根节点
    for(int i=0;i<(int)s.length();i++)
    {
        int word=s[i]-'a';	//假设是只含有小写字母的例子
        if(!trie[p][word])return 0;	//如果没有这个节点的话就直接返回
        p=trie[p][word];	//进入到这个节点
    }
    return cnt[p];	//走到了最后
}
int trie_query_prefix(string s)	//查询该字符串的前缀的数量
{
    int p=0;	//根节点
    int ans=0;  //答案
    for(int i=0;i<(int)s.length();i++)
    {
        int word=s[i]-'a';
        if(!trie[p][word])break;	//计数已经完成
        p=trie[p][word];
        ans+=cnt[p];	//加上这个前缀的数量
    }
    return ans;
}
```

### 字符串哈希

> **BKDR hash**
>
> 将字符串看成是P进制数字,P的选取是131 或13331冲突率较低
>
> 用unsigned long long 来进行隐式用$2^{64}-1$来取模

```c++
const int n,P=13331;
vector<unsigned long long> h(n,0);	//储存前i个字符串的哈希值
vector<unsigned long long> p(n,0);	//储存P的i次方
void BKDR_hash(string s)	//初始化
{
    p[0]=1;
    h[0]=s[0];	//初始化
    for(int i=1;i<n;i++)
    {
        p[i]=p[i-1]*P;	//次方
        h[i]=h[i-1]*P+s[i];	//hash值
    }
}
unsigned long long get_hash(int left,int right)
{
    return left?h[right]-h[left-1]*p[right-left+1]:h[right];	//获取区间内字符串的哈希值
}
```

## 最长回文子串

### 枚举中心点

> 对所有可能的中心点进行枚举,向两边扩展,复杂度为$O(n^2)$
>
> 也可以在可能的中心点上进行字符串哈希二分长度,复杂度为$O(nlogn)$

```c++
//返回最长回文子串长度
int expand_center(string s)
{
    int l=0,r=0,len=s.size();
    for(int i=0;i<(int)s.size()-1;i++){	//枚举中心点
        if(i>0&&s[i-1]==s[i+1]){		//奇数长度子串的中心
            int p1=i-2,p2=i+2;
            while(p1>=0&&p2<len&&s[p1]==s[p2])p1--,p2++;	//进行扩展匹配
            if(--p2-(++p1)>r-l)l=p1,r=p2;	//看下是否更新长度
        }
        if(s[i]==s[i+1]){	//偶数长度子串
            int p1=i-1,p2=i+2;
            while(p1>=0&&p2<len&&s[p1]==s[p2])p1--,p2++;
            if(--p2-(++p1)>r-l)l=p1,r=p2;
        }
    }
    return r-l+1;
}
```

### Manacher

> 复杂度$O(n)$

```c++
int Manacher(string s)
{
    int len=s.size();
    if(!len)return 0;	//特判空字符串
    vector<char> str(2*len+3);	//进行字符串预处理 把偶数长度字符串都换成奇数处理
    int idx=0;
    str[idx++]='$';str[idx++]='#';
    for(int i=0;i<len;i++){		//进行预处理
        str[idx++]=s[i];
        str[idx++]='#';
    }
    str[idx]='*';
    int len2=len*2+2;	//处理后的字符串长度
   	vector<int> p(len2,1);	//半径数组
    int c=0,r=0;	//最长回文串的中心和右边界
    for(int i=1;i<len2;i++){
        p[i]=r>i?min(p[2*c-i],r-i):1;	//判断之前的区间是否可以复用判断 一个是左边对称已知半径的长度 一个是右边界长度
        while(str[i+p[i]]==str[i-p[i]])p[i]++;	//进行暴力延展匹配
        if(i+p[i]>r)	//如果延展之后的长度比已知最长串边界要大
        {
            r=p[i]+i;	//最大半径边界修改
            c=i;		//中心更改
        }
    }
    int ans=0;
    for(int i=0;i<len2;i++)ans=max(ans,p[i]);	//找到最长的长度
    return ans-1;
}
```

## AC自动机

> **多模式匹配**
>
> 通过公共后缀来进行匹配失败时候的跳转 **因为模式串在字符串中可重叠出现**
>
> 利用了trie树和类KMP的思想 可以近似看作在trie树上的kmp匹配
>
> 复杂度$O(n)$
>
> **构建fail指针**
>
> + fail本质是当前pattern的最长后缀
>
> + 第一层的全部指向root
> + 通过BFS遍历后面的节点 **因为跳转是从长到短的**
> + 如果**当前节点x的父节点f的fail指针**拥有和当前节点**一样字符**的节点y 那么x的fail指向y
>
> **匹配**
>
> + 从根节点出发
> + 正常的trie树匹配过程 遇到节点就把以该节点为结尾的模式串添加计数到答案中
>     + 每次添加计数都要把该节点的fail链上的计数都遍历添加了
>     + 如 s:  abcde p: abcde bcde cde de e 那么在匹配到e的时候应该要通过fail链把五个都加上 
> + 匹配失败的时候 (没有了相同字符的孩子)
>     + 进行跳转 如果跳转后的有相同字符的孩子就进入它
>     + 直到进入了或者到根节点了
> + 直到字符串遍历完成

> 给定n个pattern 和 s 求s中出现了多少个pattern **(重复的不算)**
>
> patterns的字符总长为m	只有小写字母

```cpp
const int n;
const int m;	//用模式串来建树 m其实就是节点数
vector<string> patterns(n);	//模式串
string s;	//文本串
struct TrieNode{
    int son[26]={0};	//此处只有小写字母
    int cnt=0;
    int fail=-1;	//-1表示没有fail指针
};
vector<TrieNode> trie(m);	//数组模拟 预先分配空间
int idx=0;
void insert(string s)	//trie的插入 详见字典树
{
    int p=0;
    for(int i=0;i<s.size();i++)
    {
        int word=s[i]-'a';
        if(!trie[p].son[word])trie[p].son[word]=++idx;
        p=trie[p].son[word];
    }
    trie[p].cnt++;
}
void fail_pre()
{
    queue<int> q;	//BFS预处理
    for(int i=0;i<26;i++)	//先把第一层的处理掉
    	if(trie[0].son[i]){
            trie[trie[0].son[i]].fail=0;	//第一层的fail全部指向根节点
            q.push(trie[0].son[i]);	//压入队列bfs
        }
    while(q.size())
    {
        int f=q.front();	//拿出一个节点 这个节点的那层fail已处理
        q.pop();
        for(int i=0;i<26;i++)
        	if(trie[f].son[i]){	//遍历所有儿子
                int now=trie[f].son[i];	//儿子的索引
                int ffail=trie[f].fail;	//父亲的fail指向的那个节点
                while(~ffail&&!trie[ffail].son[i])ffail=trie[ffail].fail;	//终止条件为跳到了root (只有root没有fail) 或者 找到了可以进入的节点
                if(~ffail)trie[now].fail=trie[ffail].son[i];	//如果找到了 连接fail指针
                else trie[now].fail=0;	//没有找到只能指向根节点
                q.push(now);	//记得加入队列继续
            }
    }
}
int query(string s)
{
    int ans=0;
    int p=0;
    for(int i=0;i<s.size();i++)
    {
        int word=s[i]-'a';
        while(!trie[p].son[word]&&~trie[p].fail)p=trie[p].fail;	//如果匹配不到就一直跳转 直到到了根节点(只有根节点没有fail)
        if(trie[p].son[word])
            p=trie[p].son[word];	//如果匹配到了就进入节点
        else continue;	//没有匹配到那此时p肯定在root 可以去匹配下一个了文本字符了
        int p2=p;
        while(~trie[p2].fail&&~trie[p2].cnt){
            //把fail链上的全部都给加上
            ans+=trie[p2].cnt;
            trie[p2].cnt=-1;
            //题目要求 匹配过的串下次不用再匹配了
            //此处置为-1而不是0就可以在一开始就终止整条fail链的跳转
            p2=trie[p2].fail;
        }
    }
    return ans;
}
```

# 数据结构

## 哈希双向链表

> 真正意义上实现O（1）的删除和插入
>
> 通过哈希表的直接查键

```cpp
struct node{		//普普通通的双向链表结构
    int val;
    node *next,*last;
    node(int v){
        val=v;
        next=last=nullptr;
    }
};
unordered_map<int,node*> List;	//链表
List[-1]=new node(0);	//把一个特殊键值设为空头节点 prehead
void Insert(int key,int nowKey,int x){	//在key的后面插入一个key为nowKey值为x的节点
    //注意 第一个节点插入时 key为prehead的值 这里为-1
    node* newNode=new node(x);
    List[nowKey]=newNode;	//创造新的节点
    List[newNode]->next=List[key]->next;
    List[key]->next=newNode;	//链表上更改链接
}
void Delete(int key){	//删除掉key为key这个节点
    List[key]->last->next=List[key]->next;
    delete List[key];	//释放内存
}
```

## 树的简单结构

> **二叉树** *Binary Tree	BT*
>
> + 每个节点最多有两个子树

```c++
struct BT{
    type data;
    BT *left;
    BT *right;
};
```

### 满二叉树 

*Full Binary Tree	FBT*

> + 国内定义
>     + 二叉树的层数为K 那么总结点数为$2^K-1$
>     + 为金字塔型 无缺口的
> + 国际定义
>     + 二叉树的子节点要么为0 要么为2
>         + 国际把国内定义的满二叉树叫做 **完美二叉树** *Perfect Binary Tree	PBT*

### 完全二叉树

*Complete Binary Tree	CBT*

> + 叶子结点只可能在最深的两层出现
> + 子节点数为1的节点只有1个或没有
> + 1-n的节点与同样深度的满二叉树的1-n节点相互对应
>
> + 节点从1-n编号 如果i>1 那么该节点的父节点为i/2
> + i>1 奇数节点是右节点 偶数节点是左节点
> + 节点i的左孩子节点为2*i 右孩子节点为2*i+1
> + i>n/2的节点均为叶子节点

### 二叉查找树

*Binary Sort Tree	BST*

> + 若左子树不为空 那么左子树上节点的值均小于根节点
> + 若右子树不为空 那么左子树上节点的值均大于根节点
> + 左右子树也为二叉查找树
> + 没有值相同的节点
> + 中序遍历的结果是排序

```c++
//BST的查找 失败返回false
bool BST_find(TreeNode* t,int v)
{
    if(!t)return false;
    if(t->val==v)return true;
    if(v<t->val)return BST_find(t->left);
    if(v>t->val)return BST_find(t->right);
}
//BST的插入
TreeNode* BST_INS(TreeNode* t,int v)
{
    if(!t)return TreeNode(v);	//空节点直接链接
    if(v<t->val)t->left=BST_INS(t->left,v);		//要插入的节点肯定在左边
    else if(v>t->val)t->right=BST_INS(t->right,v);		//要插入的节点肯定在右边
    return t;
}
//BST的删除
void BST_DEL(TreeNode* t,int v)
{
    if(!t)return nullptr;
    if(v<t->val)BST_DEL(t->left,v);
    else if(v>t->val)BST_DEL(t->right,v);
    else{
        //找到被删的节点
        //不考虑空间的释放
        //考虑空间的释放用递归返回节点或传入父亲节点做法
        if(!t->left)*t=*(t->right);	//只有右节点 包含左右都为空的情况
        else if(!t->right)*t=*(t->left);	//只有左节点
        else {
            //找到右边的最小节点
            TreeNode* p=t;
            TreeNode* x=p->right;	//肯定会有
            while(x->left){
                p=x;
                x=x->left;
            }
            //循环结束时 p是x的父亲 x是最小节点
            t->val=x->val;	//赋值后继节点
            p->left=x->right;	//把节点删掉
        }
    }
}
```

### 线索二叉树

*Threaded Binary Tree	TBT*

> 以某种遍历方式 在节点上利用空指针域来储存前驱或后继

```c++
//线索二叉树的结构
struct TBT{
    type data;
    TBT *left;
    TBT *right;
    bool lTag;
    bool rTag;
    //tag为0时表示正常的左右子节点 为0时表示前驱或后继
};
```

> 通过一次的遍历就能将其线索化

```c++
TreeNode* pre;	//全局指针 始终指向刚刚访问过的节点
void in_threading(TreeNode* t)		//以中序遍历为例
{
    if(!t)return;
    if(!t->left){
        t->lTag=1;
        t->left=pre;	//指向前驱;
    }
    else in_threading(t->left);
    if(!t->right){
        pre->rTag=1;
        pre->right=t;
    }
    pre=t;	//前驱处理完毕
    in_threading(t->right);	//注意和上面语句的顺序
}
```

## 哈夫曼树

### 建树

> 又称**最优二叉树** 是带权路径长度最短的二叉树
>
> **带权路径** 
>
> + 假设某个节点的权值为w 该节点的深度为h 那么带权路径长度为wh
>
> + 总的带权路径就是**所有叶子节点带权路径的和**或者是**所有节点的权值之和**
>
> **建树过程**
>
> + 每次把节点权值中**最小**的**两个节点**连接到一个新的节点的左右孩子上 其中新节点的权值为**两个节点之和**
> + 采用最小堆来维护最小的节点
> + 合并操作后把左右孩子的权值从堆中pop掉然后把新节点权值push进堆
> + 直到堆中只有一个数字了

```cpp
struct TreeNode {	//树节点的定义
	int c;
	int w;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int c_,int w_, TreeNode* l, TreeNode* r)
	{
		this->c = c_;
		this->w = w_;
		left = l;
		right = r;
	}
	bool operator > (const TreeNode &t) const	//对节点重载<可以用stl
	{
		return w > t.w;
	}
};

TreeNode* buildHuffman(vector<pair<int,int>> &arr)	//传入叶子节点的权值
{
	priority_queue<TreeNode, vector<TreeNode>, greater<TreeNode>> heap;	//最小堆
	for (int i = 0; i < arr.size(); i++) {
		TreeNode t(arr[i].first,arr[i].second, 0, 0);
		heap.push(t);
	}
	while (heap.size() > 1)	//当还能把森林or叶子拼成树时
	{
		TreeNode* t1 = new TreeNode(heap.top().c,heap.top().w, heap.top().left, heap.top().right);
		heap.pop();
		TreeNode* t2 = new TreeNode(heap.top().c, heap.top().w, heap.top().left, heap.top().right);
		heap.pop();
		TreeNode* t = new TreeNode(' ',t1->w + t2->w, t1, t2);	//拿出最小的两个节点拼成新的
		heap.push(*t);
	}
	return new TreeNode(heap.top().c,heap.top().w, heap.top().left, heap.top().right);
}
```

### 哈夫曼编码

> + 哈夫曼树中只有叶子节点有意义 左子节点的边记作0 右子节点的边记作1 那么从根节点到某个叶子结点的边的编码就是该叶子节点的哈夫曼编码
> + 哈夫曼编码是前缀编码 任一个叶子结点的编码都不是另一个叶子编码的前缀

```cpp
map<char,pair<int, string>> codes;	//表示叶子c权值为i 它的编码为j 元组(i,j)数组
pair<int, string> tmp;	//全局保存临时路径 便于回溯
void get_code_dfs(TreeNode* t)
{
	if (!t->left && !t->left) {	//叶子节点
		tmp.first = t->w;
		codes[t->c]=tmp;
	}
	else {
		if (t->left) {
			tmp.second.push_back('0');	//左0
			get_code_dfs(t->left);
		}
		if (t->right) {
			tmp.second.push_back('1');	//右1
			get_code_dfs(t->right);
		}
	}
	tmp.second.pop_back();	//回溯还原
}
```

### 哈夫曼译码

> 按照同一棵树进行译码
>
> 还是按照左0右1

```cpp
string res_code;
void translate_code_dfs(string &code,int idx,TreeNode* t)
{
	if (!t)return;
	if (idx >= code.size())return;
	if (!t->left&&!t->right) {	//到了叶子节点
		res_code.push_back(t->c);
		translate_code_dfs(code, idx, root);	//重新重头开始
		return;
	}
	if (code[idx] == '0')translate_code_dfs(code, idx + 1, t->left);	//左0
	else translate_code_dfs(code, idx + 1, t->right);	//右1
}
```

## 单调队列

> **滑动窗口问题**
>
> 有一个大小为 *k* 的滑动窗口从数组的最左侧移动到数组的最右侧,只可以看到在滑动窗口内的 *k* 个数字,滑动窗口每次只向右移动一位.求滑动窗口中的最大值.

> deque实现

```c++
vector<int> arr;
int k;
vector<int> window_max()
{
    vector<int> ans;	//存每个窗口的最大值
    deque<int> q;	//q存的是索引 保证队列中不会超出k范围
    for(int i=0;i<k-1;i++){
        while(q.size()&&arr[q.back()]<=arr[i])q.pop_back();//进队时保证前方的都比他大
        q.push_back(i);
    }
    for(int i=k-1;i<(int)arr.size();i++){
        if(i-q.front()==k)q.pop_front();	//超出k个 队头出队
        while(q.size()&&arr[q.back()]<=arr[i])q.pop_back();
        q.push_back(i);
        ans.push_back(arr[q.front()]);
    }
    return ans;
}
```

> 数组实现

```c++
vector<int> arr,q;
int k,front=0,back=0;	//当front=back时队列为空
vector<int> window_max()
{
    vector<int> ans;
    for(int i=0;i<k-1;i++){
        while(back>front&&arr[q[back-1]]<=arr[i])back--;	//队列不空
        q[back++]=i;
    }
    for(int i=k-1;i<(int)arr.size();i++){
        if(i-q[front]==k)front++;	//队头出队 队肯定不为空
        while(back>front&&arr[q[back-1]]<=arr[i])back--;
        q[back++]=i;
        ans.push_back(arr[q[front]]);
    }
    return ans;
}
```

## 单调栈

> 栈内是单调的,用来解决如:
>
> 给定一个长度为N的整数数列,输出每个数左边第一个比它小的数,如果不存在则输出-1.

```c++
vector<int> arr;
vector<int> m_stack()
{
    vector<int> ans;
    stack<int> stack_;
    for(int i=0;i<(int)arr.size();i++){
        if(stack_.empty())ans.push_back(-1);	//当前已经是最小的数了
        else {
            while(stack_.size()&&stack_.top()>=arr[i])stack_.pop();//找到第一个比当前小的
            if(stack_.empty())ans.push_back(-1);
            else ans.push_back(stack_.top());
            stack_.push(arr[i]);
        }
    }
    return ans;
}
```

## 并查集

### 朴素并查集

> 最坏情况当树退化成链的时候 每次的查询和合并操作都是$O(n)$的

```c++
int n;
vector<int> f(n);	//father节点
void init()	//初始化
{
    for(int i=0;i<n;i++)
        f[i]=i;		//表示每个节点是独立的 他的祖先节点是自己
}
int find(int x)
{
    return f[x]==x?x:find(f[x]);	//如果该节点是祖先节点 那么就返回 否则递归往上寻找
}
void Union(int a,int b)
{
    f[find(a)]=find(b);	//令a的祖先连到b的祖先	相同祖先的话会自己连自己
}
```

### 高度并查集

> 额外储存树的高度,每次合并低的树指向高的树,控制树高增加
>
> 查找和合并操作都是$O(logn)$

```c++
int n;
vector<int> f(n);	//father 节点
vector<int> h(n);	//储存树的高度且仅祖先节点有意义
void init()	//初始化
{
    for(int i=0;i<n;i++){
        f[i]=i;
        h[i]=1;		//初始每棵树的高度都为1
    }
}
int find(int x)
{
    return f[x]==x?x:find(f[x]);	//如果该节点是祖先节点 那么就返回 否则递归往上寻找
}
void Union(int a,int b)
{
    int fa=find(a),fb=find(b);
    if(fa==fb)return;	//相同祖先 不进行其他操作
    if(h[fa]<=h[fb]){
        f[fa]=fb;
        h[fb]=max(h[fb],1+h[fa]);	//当树高相等时 合并需要把树高度+1
    }
    else f[fb]=fa;	//此时高度合并后的肯定不会超出h[fa]
}
```

### 维护集合大小的路径压缩并查集

> 查询和合并操作均为$O(1)$

```c++
int n;
vector<int> f(n);	//forefather 节点
vector<int> h(n);	//储存树的高度且仅祖先节点有意义
vector<int> size(n);	//储存树的节点数量且仅祖先节点有意义
void init()
{
	for(int i=0;i<n;i++){
        f[i]=i;
        size[i]=1;		//初始每棵树的大小都为1
        h[i]=1;		//初始每棵树的高度都为1
    }
}
int find(int x)
{
    return f[x]=(f[x]==x?x:find(f[x]));	//直接把当前节点的父节点改为祖先节点
}
void Union(int a,int b)
{
    int fa=find(a),fb=find(b);
    if(fa==fb)return;	//相同祖先 不进行其他操作
    if(h[fa]<=h[fb]){
        size[fb]+=size[fa];
        f[fa]=fb;
        h[fb]=max(h[fb],1+h[fa]);	//当树高相等时 合并需要把树高度+1
    }
    else {
        size[fa]+=size[fb];
        f[fb]=fa;	//此时高度合并后的肯定不会超出h[fa]
    }
}
```

### 维护链长的路径压缩并查集

> 维护元素到根节点的长度	(抽象成为一条链  与实际的树不一样)
>
> 合并的时候把原来的根的链长从0加上要移到的集合的大小 (即抽象链 实际上是以树直接接到另一个集合的根的)
>
> 查询的时候 每个元素都加上其祖先节点的链长即可 (一定要先往祖先节点搜索完)

```cpp
vector<int> f,dis,cnt;	//祖先 到根的距离 集合的大小
int find(int x){
    if(f[x]==x)return x;
    int root=find(f[x]);
    dis[x]+=dis[f[x]];	//这时候x的祖先的长度已经算好了
    return f[x]=root;
}
void Union(int a,int b){
    int fa=find(a),fb=find(b);
    if(fa==fb)return;
    dis[fa]=cnt[fb];	//加上另一个集合大小 原来作为根是0
    cnt[fb]+=cnt[fa];	//改掉集合大小 不能和上面语句顺序交换
    f[fa]=fb;	//合并
}
```

## 树状数组

> 一种进行动态维护前缀和的结构

### 区间查询和单点修改

```c++
int n;
vector<int> tree;	//树状数组
vector<int> arr;	//原数组
//树状数组下标从1开始 0没有意义
void add(int x,int c){		//单点修改 在x位置上+c
    while(x<=n){
        tree[x]+=c;
        x+=lowbit(x);	//在修改的时候不断往父节点修改
    }
}
int query(int x){		//查询x的位置的前缀和
	int ans=0;
    while(x>0){
        ans+=tree[x];
        x-=lowbit(x);	//跳转到左边区域
    }
    return ans;
}
inline int query(int l,int r){	//查询区间和
    return query(r)-query(l-1);
}
inline void init(){	//初始化树状数组
    for(int i=1;i<(int)arr.size();i++)add(i,arr[i]);
}
```

### 区间修改和单点查询

> 实际上是差分数组的应用

```c++
int n;
vector<int> tree;	//树状数组
vector<int> arr;	//原数组
//树状数组下标从1开始 0没有意义
void add(int x,int c){		//单点修改 在x位置上+c
    while(x<=n){
        tree[x]+=c;
        x+=lowbit(x);	//在修改的时候不断往父节点修改
    }
}
void add(int l,int r,int c){	//区间修改
    add(l,c);		//差分操作
    add(r+1,-c);
}
int query(int x){	//单点查询实际上就是调用了求差分数组的前缀和
	int ans=0;
    while(x>0){
        ans+=tree[x];
        x-=lowbit(x);
    }
    return ans;
}
inline void init(){
    for(int i=1;i<(int)arr.size();i++){
        int d=arr[i]-arr[i-1];
        add(i,d);		//差分的初始数组
    }
}
```

### 区间修改和区间查询

> 根据公式,$\sum_{i=1}^x arr_i=\sum_{i=1}^x \sum_{j=1}^i d_j=(x+1)\sum_{i=1}^x d_i-\sum_{i=1}^xid_i$,即差分前缀和的前缀和,只需要维护两个树状数组$\sum_{i=1}^xd_i$和$\sum_{i=1}^xid_i$即可

```c++
int n;
vector<int> tree;	//树状数组
vector<long long> tree2;	//辅助树状数组
vector<int> arr;	//原数组
//树状数组下标从1开始 0没有意义
void add(int x,int c){		//单点修改 在x位置上+c
    int i=x;
    while(i<=n){
        tree[i]+=c;		//正常的修改
        tree2[i]+=1ll*x*c;	//辅助数组修改
        i+=lowbit(i);	//在修改的时候不断往父节点修改
    }
}
void add(int l,int r,int c){	//区间修改
    add(l,c);		//差分操作
    add(r+1,-c);
}
long long query(int x){		//x位置的前缀和查询
	long long ans=0;
    int i=x;
    while(i>0){
        ans+=1ll*(x+1)*tree[i]-tree2[i];	//根据公式
        i-=lowbit(i);
    }
    return ans;
}
inline long long query(int l,int r){	//查询区间和
    return query(r)-query(l-1);
}
inline void init(){
    for(int i=1;i<(int)arr.size();i++){
        int d=arr[i]-arr[i-1];
        add(i,d);		//差分的初始数组
    }
}
```

## ST表

> ST表用来处理一类**静态**区间问题,只要该区间符合性质: $f(L,R)=f(f(L,a),f(b,R))\;\;(a>=b)$
>
> 可知此处ab部分区间是可以有重叠的
>
> 函数max,min,gcd,lcm等这些均符合该性质 即$f(a,a)=a$
>
> 只能解决静态的问题 
>
> 利用了区间dp的思想 

> **RMQ**区间最值问题
>
> $O(nlogn)$预处理  $O(1)$询问
>
> $dp_{i,j}$定义为左端点为$i$,长度为$2^j$的区间所求值,也就是区间$[i,i+2^j-1]$
>
> 状态转移方程为:$dp_{i,j}=max(dp_{i,j-1},dp_{i+2^{j-1},j-1})$

```c++
vector<int> arr;	//数组下标从1开始方便处理
vector<vector<int>> dp;
int n;
void pre()
{
	for (int i = 1; i <= n; i++)dp[i][0] = arr[i];		//初始化 表示长度为1的时候是自身
	for (int j = 1; j <= log2(n); j++)		//枚举区间长度j
		for (int i = 1; i + (1 << j) - 1 <= n; i++)	//枚举起点i
			dp[i][j] = max(dp[i][j - 1], dp[i + (1 << (j - 1))][j - 1]);
}
int query(int l, int r)
{
	int len = log2(r - l + 1);	//保证区间重叠且不超过l r
	return max(dp[l][len], dp[r - (1 << len) + 1][len]);
}
```

## 对顶堆

> 对顶堆可以动态维护中位数和动态第K大/小值等问题
>
> 本质是维护了一个大顶堆和一个小顶堆
>
> + 两个堆各自是单调的
> + 大顶堆在下,小顶堆在上
> + 保证小顶堆里面的元素都比大顶堆的要大
> + 堆中元素只能不断动态增加

> **动态中位数**

```c++
priority_queue<int> big;	//大顶堆
priority_queue<int,vector<int>,greater<int>> small;	//小顶堆
inline void add(int x)
{
    if(small.empty()||x>small.top())small.push(x);	//如果上面的小堆是空或者元素比分界线大
    else big.push(x);								//先加入小堆 否则加入下面
    //开始调整 保证上方的小堆大小大于等于下面大堆的大小 保证从上方取中位数
    if((int)big.size()-(int)small.size()==1){		//下面比上面多了一个
        small.push(big.top());	//把下面的移到上面去
        big.pop();
    }
    else if((int)small.size()-(int)big.size()>1){	//上面的堆数量多了
        big.push(small.top());
        small.pop();
    }
}
```

> **动态第K小数**

```c++
int K=0;	//该K值先获取最值再增加 保证小顶堆恒有元素 获取时候保证下面的大堆的大小为K-1
priority_queue<int> big;	//大顶堆
priority_queue<int,vector<int>,greater<int>> small;	//小顶堆
inline void add(int x)	//动态添加元素	此时无需调整
{
	if (small.empty() || x > small.top())small.push(x);
	else big.push(x);
}
void adjust() {		//调整至下面的大堆大小为K-1
	while (big.size() < K&&small.size()) {
		big.push(small.top());
		small.pop();
	}
	while (big.size() > K&&big.size()) {
		small.push(big.top());
		big.pop();
	}
}
//数据应当保证adjust时元素数大于K
```

## 树的dfs序

> 用dfs先序的方法把从根出发对树进行遍历, 并且记录下进入某个节点x的时间in[x]以及记录某个节点遍历完所有子节点后的离开时间out[x]. 从时间的递增性质可以知道对于节点x的(in[x],out[x])是可以构成一个合法区间的,且该范围的含义是**包含x节点与其所有子节点的区间**,x节点在区间的最左端,这样就可以实现对树的所有子节点操作改为对线性序列的区间操作.
>
> 显然,根据in[x]可以给树的节点进行编号并构成一个树的序列

```c++
//假设该树是以二维数组储存的	tree[x][i] 表示当前节点x的第i个儿子是哪个节点
vector<vector<int>> tree;
//range[x] 表示节点x的子节点们的范围(range[x].first,range[x].second)
//其中range[x].first为x节点的序号
vector<pair<int,int>> range;
vector<int> seq;
void dfs(int x)
{
    seq.push_back(x);
    range.first = seq.size();
    for(int i=0;i<tree[x].size();i++)
        dfs(tree[x][i]);
    range[x].second = seq.size();
}
```

## 线段树

### 基础线段树

> $O(logn)$单点修改	$O(logn)区间查询$
>
> 无lazy标记 递归版

```c++
//node的传值永远从根节点开始传	因为是递归写法
const int n;	//数组的大小
const int h;	//树的高度 为log2(n)+1
vector<int> arr(n);	//举例下标从0开始
vector<int> tree(2<<h-1);	//用数组来模拟树结构	化简之后约为n的四倍
void build(int node,int start,int end)
{
    if(start==end){
        tree[node]=arr[start];	//边界只有一个点
        return;
    }
    int left_node=(node<<1)+1;
    int right_node=(node<<1)+2;
    int mid=(start+end)>>1;		//分段
    build(left_node,start,mid);		//递归处理左边的
    build(right_node,mid+1,end);	//递归处理右边的
    tree[node]=tree[left_node]+tree[right_node];	//求总和 可以是其他操作
}
//单点在arr[idx]上改为val
void update(int node,int start,int end,int idx,int val)
{
   	if(start==end){
        arr[idx]=val;	 //可以是其他操作
        tree[node]=val;	 //可以是其他操作
        return;
    }
    //熟悉的获取左右子节点和中点
    int left_node=(node<<1)+1;
    int right_node=(node<<1)+2;
    int mid=(start+end)>>1;
    if(idx<=mid)update(left_node,start,mid,idx,val);	//要修改的点在左边
    else update(right_node,mid+1,end,idx,val);	//右边
    tree[node]=tree[left_node]+tree[right_node];	//更新时候全部更新总和 可以是其他操作
}
int query(int node,int start,int end,int L,int R)
{
    if(L>end||R<start)return 0;	//完全不在需求区间内
    if(L<=start&&end<=R)return tree[node];	//该线段完全包含在需求区间内
    //熟悉的获取左右子节点和中点
    int left_node=(node<<1)+1;
    int right_node=(node<<1)+2;
    int mid=(start+end)>>1;
    //此处L和R没必要修改
    int sum_left=query(left_node,start,mid,L,R);	//求左边
    int sum_right=query(right_node,mid+1,end,L,R);	//求右边
    return sum_left+sum_right;	//可以是其他操作
}
```

### 区间更新线段树

> 操作为加和乘的示例
>
> 询问为求和的示例
>
> 两个lazy标记

```c++
const int n;
vector<int> arr(n);	//arr下标从1开始方便处理
struct{
    int L,R;
    int add_lazy=0;	//加的懒惰标记 初始化为0
    int mul_lazy=1;	//乘的懒惰标记 初始化为1
    int sum=0;
}tree[4*n];		//公式化简后约为4倍空间
#define now (tree[node])	//当前节点
#define Lidx (node<<1)		//左子节点下标
#define Ridx (node<<1|1)		//右子节点下标
#define Lnode (tree[Lidx])		//左子节点
#define Rnode (tree[Ridx])		//右子节点	
void build(int node,int L,int R)
{
    now.L=L;
    now.R=R;
    if(L==R){	//到达叶子结点
        now.sum=arr[L];
        return;
    }
    int mid=(L+R)>>1;
    build(Lidx,L,mid);	//分割左子节点建树
    build(Ridx,mid+1,R);	//分割右子节点
    now.sum=Lnode.sum+Rnode.sum;	//递归把下面的和传上来
}
void set_lazy(int node,int add_v,int mul_v)
{
    //对子节点都进行加add_v操作等于该节点的sum加上子节点长度乘上add_v
    //对子节点都进行乘mul_v操作相当于该节点的sum直接乘上mul_v
    now.sum=now.sum*mul_v+(now.R-now.L+1)*add_v;
    //加的标记除了直接标记add_v还需要在进行乘法的时把之前标记的add_v进行乘
    now.add_lazy=now.add_lazy*mul_v+add_v;
    //乘的标记
    now.mul_lazy*=mul_v;
}
void push_down(int node)
{
    if(now.L==now.R)return;	//叶子节点不需要下放
    if(!now.add_lazy&&now.mul_lazy==1)return;	//无标记
    //左右下放标记
    set_lazy(Lidx,now.add_lazy,now.mul_lazy);
    set_lazy(Ridx,now.add_lazy,now.mul_lazy);
    //标记还原
    now.add_lazy=0;
    now.mul_lazy=1;
}
//只能进行单独加或者单独乘
//加时mul_v传参为1 乘时add_v传参为0
void update(int node,int L,int R,int add_v,int mul_v)
{
    if(L<=now.L&&now.R<=R){		//更新到想要的了
        set_lazy(node,add_v,mul_v);	//直接设置懒惰标记 下面的子树不再遍历
        return;
    }
    push_down(node);	//需要进行下放标记了
    int mid=(now.L+now.R)>>1;
    if(L<=mid)update(Lidx,L,R,add_v,mul_v);	//更新的区间包含左子节点区间
    if(mid<R)update(Ridx,L,R,add_v,mul_v);	//更新的区间包含右子节点区间
    now.sum=Lnode.sum+Rnode.sum;
}
int query(int node,int L,int R)
{
    if(L>now.R||now.L>R)return 0;	//超出所求范围
    if(L<=now.L&&now.R<=R)return now.sum;	//找到要的完整被包含的区间
    push_down(node);	//没找到 随着询问下放标记
    return query(Lidx,L,R)+query(Ridx,L,R);
}
```

# 几何

## 判断圆和矩形是否重叠

> + 先把矩形中心移到坐标原点 同时把圆也相对移动
> + 把圆移到第一象限 因为此时矩形中心在原点 所以无论圆在哪个象限都能映射到第一象限进行相交判断
> + 计算出原点与矩形右上角的dx与dy
> + dx与dy如果小于0 那么则直接映射为0 因为小于0时dx与dy方向上的可当做圆心垂直矩形右上角最短 即只需要考虑大于0的方向
> + 计算dx与dy和矩形右上角的欧拉距离
> + 与半径进行对比

```c++
//圆半径和原点坐标 矩形左下角坐标和右上角坐标
bool checkOverlap(double r,double x,double y,double x1,double y1,double x2,double y2)
{
    double x_center = (x1+x2)/2, y_center = (y1+y2)/2;
    x2-=x_center;
    y2-=y_center;
    x = fabs(x-x_center);
    y = fabs(y-y_center);
    double dx = max(x-x2,0.0), dy = max(y-y2,0.0);
    return dx*dx+dy*dy <= r*r;
}
```

## 圆覆盖最多点问题

> 给定一个**半径为r**的圆
>
> 给定一些**点的坐标($x_i,y_i$)**
>
> 求圆能覆盖最多点的个数

> **暴力枚举圆**
>
> + 最优解的圆是可移动的
> + 必定可以移动到某两个或以上点在最优圆的边缘上
> + 枚举这两个点在固定半径圆边缘上的圆心位置
> + 暴力循环看在这两个点生成的圆有多少个点在里面
> + 计算圆心位置根据勾股定律和三角函数可算得

```cpp
struct Point{
    double x,y;
    Point(double x_,double y){x=x_;y=y_;}
};
vector<Point> points;
int r;	//半径
int n;	//点数
inline double dist(Point a,Point b)
{
    return sqrt(pow(a.x-b.x,2)+pow(a.y-b.y,2));	//欧几里得距离
}
Point getCenter(Point a,Point b)	//重点
{
    Point mid=Point((a.x+b.x)/2,(a.y+b.y)/2);	//中点
    double theta=atan2(a.x-b.x,b.y-a.y);	//把y分量向量反向 模板背就行
    double d=sqrt(r*r-pow(dist(a,mid),2));	//ab直线和圆心的距离
    return Point(mid.x+d*cos(theta),mid.y+d*sin(theta));
}
int numPoints()
{
    int ans=1;	//最少有一个点
    for(int i=0;i<n;i++)	//遍历两个相同的点时由于顺序不同 会把两个方向的圆都考虑
        for(int j=0;j<n;j++)
        {
            if(dist(points[i],points[j])>2.0*r)continue;	//不可能
            Point center=getCenter(points[i],points[j]);
            int cnt=0;
            for(int k=0;k<n;k++)	//暴力计数
                if(dist(center,points[k])<=1.0*r+1e-8)cnt++;	//注意误差
            ans=max(ans,cnt);
        }
    return ans;
}
```

# 博弈论

## ICG游戏

经典的博弈论游戏Impartial Combinatorial Games

**游戏定义**

+ 玩家只有两个人
+ 游戏给定了状态之间转移的规则
+ 玩家的状态有限
+  每个人轮流移动
+ 有明确的状态结束情况

**状态定义**

+ 必败点 当前玩家无论怎么移动都**只能**转移到必胜点
+ 必胜点 当前玩家**存在**一种可以移动到必败点的方法
+ 最终(结束)点 为必败点

## Bash 博弈

> 只有一堆n个物品 两个人轮流从这对物品取物 规定每次至少取一个 最多取m个 最后取光者胜

显然 如果n=m+1 由于一次最多只能取m个 所以无论先取者拿走多少个 后取者都能够一次拿走剩余的物品

当n=k(m+1)+x 时 先取者只要保证拿走x个 就能把局面变成 n=k(m+1) 无论后取者拿走多少个 都会重新把局面变回到n=(k-1)(m+1)+k' 那么最终显然会让后手变成局面n=m+1 这是必输的局面

## 威佐夫博弈

> 有两堆若干个物品 两个人轮流从某一堆或同时从两堆中取同样任意多的物品 规定每次最少取一个 最后取光获胜

假设两堆物品的数量为 (n,m)  那么该游戏的前几个必败点可枚举出为:

(0,0)  (1,2)  (3,5)  (4,7)  (6,10)  (8,13)  (9,15)  (11,18)  (12,20) ......

从中找到规律为 第k个必败点为 (x,x+k)  k从0开始  x为前面必败点两堆数量中没出现过的最小自然数

**胜负手交换**

可以证明出  当在必败点的时候 肯定会转移到必胜点   因为假设只改变一堆物品的数量  那么另一个数量固定  这样的组合肯定不存在于必败点中  而假设改变了两队物品的数量 但是由于其差值不变 那么也比定不存在这样的组合在必败点中

也可以证明出 在必胜点 必定存在一种方式转移到必败点  证略

 **结论**

假设第一堆数目小于第二堆的数目 当`n==int((m-n)*(sqrt(5)+1)/2)`  (n,m)为必败点

## NIM石子游戏

> 有若干堆石子 每堆石子的数量都是有限的 有两个人 每一次操作可以从任意一堆石子中取出任意数量的石子 至少取一颗 至多取完一整堆 (不能跨堆)  两个人轮流行动 取走最后一个石子的人胜利  假设两人都足够聪明 问谁会赢

结论为 当所有石子堆的异或和为0时 当前选手必输

证明:  

当石子堆的异或和**等于k**的时候(k!=0) 那么肯定存在一堆石子X它的二进制的最高位和k的二进制最高位是同一位(否则k的最高位就无法得到)  那么此时只要把X拿剩下X^k时(最高位从1变0  数字大小肯定是变小的了)  之后的石子堆的异或和就会为0

当局面异或和为0的时候 肯定不存在一种删除方式使异或和仍为0  因为异或显然要异或自身才能变为0  假如删除了某一堆 那么原来的堆里的某两个相同数量的异或为0就会被打破  变成异或为非0的

那么到最后 显然可以取到最后一块石头的状态异或0肯定不为0 (因为是只剩下单组了)  因为全0是必败点

```cpp
vector<int> stones(n);
bool nim(){
    int ans=0;
    for(int i=0;i<n;i++)
        ans^=stones[i];
    return ans;
}
```

## SG函数

可以解决多个不同的组合博弈游戏的问题

### mex运算

定义mex(minimal excludant)为一个集合运算   mex(S) 表示最小的不属于这个集合的非负数

如 mex({0,1,2,4})=3  mex({2,3,5})=0

该运算主要是用来作必胜点和必败点之间的转移

### SG数组

SG数组是对于 **单个游戏而言的**

SG[i] 表示的是一个游戏的**i状态** (状态可以是一个值  也可以是某个集合 视游戏而定) 时候的胜负转移情况

转移方程为:

+ `SG[i] = mex(S)`    S表示为i状态可以**单步**转移的**所有**状态的**SG值**的**集合** 即若i可以单步转换到a,b,c 的话 那么SG[i] = mex({SG[a],SG[b],SG[c]})

+ `SG[i]=0`  当且仅当i为必败状态时

+ `SG[end]=0` 即结束态SG值为0

```cpp
int n;	//n表示该游戏有限的状态数
vector<int> SG(n+1,0);
vector<int> f(m);	//每一步能如何转换 m表示有多少种转换方式
void getSG(){
    SG[0]=0;	//结束状态 不同游戏不一样
    for(int i=1;i<=n;i++){	//不一定是这样的顺序转移状态 视游戏而定
        vector<bool> S(n+1,false);	//单步能转移的所有状态的SG值集合
        for(int j=0;f[j]<=i&&j<m;j++)	//遍历所有单步能转移的方式 然后填充S
            S[SG[i-f[j]]]=true;		//i-f[j]表示转移后的状态值 视不同游戏而定 不一定是减
        for(int j=0;;j++)	//其实就是计算mex函数
            if(!S[j]){
            	SG[i]=j;
            	break;
        	}
    }
}
```

### Sprague-Grundy定理

游戏和的SG值等于**各个游戏的SG值**的NIM和

```cpp
int game1_status,game2_status,game3_status;	//假如有多个游戏的多个初始态	不一定为int类型的状态

//针对不同游戏获取对应的SG值
getSG1();
getSG2();
getSG3();

//其组合博弈游戏的答案就是多个游戏的SG值的NIM和
int ans=SG1[game1_status]^SG2[game2_status]^SG3[game3_status];
```

## 树上删边游戏

> 有两个人轮流操作, 每次从一棵树上删除一条边,同时删去所有在删除边后不再和根相连的部分. 无法再进行操作者判定为失败. 一个游戏有多棵树.

### 竹子树

引入竹子树的概念 也就是一条链 很显然 当从竹子中砍掉某一节(某条边)之后  其上面的边都会一同删掉 (重力的原因)  因此 这样就等同于你可以一次对一棵竹子树删除任意节

显然 这样就可以转化为NIM游戏了

### 克朗定理

对于树上的某个点 它的分支可以转换为以这个点为根的一棵竹子树  这棵竹子树的高度等于它各个竹子树分支边高度的异或和 (显然  根据这个定理 非竹子树边也能转换为竹子树边)

那么最终 树都能转换为竹子树 也就是可以用NIM游戏来解决

其实质是 对于博弈论游戏的大部分问题  只要SG值相同 模型都可以相互转换的

## 无向图删边游戏

> 有一个无向连通图 有一个点作为图的根 两人轮流从图中删去一条边 不与根节点相连的部分移走
>
> 该图会有环 但是保证环没有共用边

### 费森定理

图中的简单环可以如下处理:

+ 偶长度环可以缩成一个新点

+ 奇长度环可以缩成一个新点加一个新边

这样的改动不会影响图的SG值

# 其他

## 前缀和

### 一维前缀和

> 求区间L-R的区间和

```c++
const int n;
vector<int> arr(n,0),sum(n,0);
void pre_sum()
{
    sum[0]=arr[0];	//第一个直接赋值
    for(int i=1;i<n;i++)sum[i]=sum[i-1]+arr[i];	//当前前缀和等于上一个前缀和加当前的数
}
#define get_sum(L,R) (L?sum[R]-sum[L-1]:sum[R])	//获取区间前缀和
```

### 二维前缀和

> 求矩形区间(x1,y1)到(x2,y2)的区间和

```c++
const int n,m;
vector<vector<int>> arr(n,vector<int>(m,0)),sum(n,vector<int>(m,0));
void pre_sum()
{
    sum[0][0]=arr[0][0];		//初始化第一个
    for(int i=1;i<m;i++)sum[0][i]=sum[0][i-1]+arr[0][i];		//初始化第一行
    for(int i=1;i<n;i++)sum[i][0]=sum[i-1][0]+arr[i][0];		//初始化第一列
    for(int i=1;i<n;i++)for(int j=1;j<m;j++)
        sum[i][j]=arr[i][j]+sum[i-1][j]+sum[i][j-1]-sum[i-1][j-1];	//矩形上部分和左部分减去重复的
}
int get_sum(int x1,int y1,int x2,int y2)
{
    if(!x1&&!y1)return sum[x2][y2];		//当左上角点为(0,0)时候
    if(!x1)return sum[x2][y2]-sum[x2][y1-1];	//只需要减去左边的前缀和
    if(!y1)return sum[x2][y2]-sum[x1-1][y2];	//只需要减去上边的前缀和
    return sum[x2][y2]-sum[x1-1][y2]-sum[x2][y1-1]+sum[x1-1][y1-1];
}
```

## 差分

### 一维差分

> 对区间L-R的数进行多次加减操作

```c++
const int n;
vector<int> arr(n,0),d(n+1,0);		//会涉及到R+1 所以空间大小+1
void difference(int L,int R,int num)
{
    d[L]+=num;
    d[R+1]-=num;
}
void deal()		//把操作映射到原数组上
{
    int add=0;	//累加项
    for(int i=0;i<n;i++)
    {
        add+=d[i];	//累加项加上区间操作
        arr[i]+=add;	//原数组更改
    }
    d=vector<int>(n+1,0);	//差分标记还原
}
```

### 二维差分

> 对矩形(x1,y1)到(x2,y2)进行多次操作

```c++
const int n,m;
vector<vector<int>> arr(n,vector<int>(m,0)),d(n+1,vector<int>(m+1,0));		//空间大小增大边界
void difference(int x1,int y1,int x2,int y2,int num)
{
    d[x1][y1]+=num;
    d[x2+1][y2+1]+=num;
    d[x1][y2+1]-=num;
    d[x2+1][y1]-=num;
}
void deal()		//把多次操作映射到原数组上
{
    vector<vector<int>> sum(n,vector<int>(m,0));	//对标记数组进行二维前缀和
    sum[0][0]=d[0][0];		//初始化第一个
    for(int i=1;i<m;i++)sum[0][i]=sum[0][i-1]+d[0][i];		//初始化第一行
    for(int i=1;i<n;i++)sum[i][0]=sum[i-1][0]+d[i][0];		//初始化第一列
    for(int i=1;i<n;i++)for(int j=1;j<m;j++)
        sum[i][j]=d[i][j]+sum[i-1][j]+sum[i][j-1]-sum[i-1][j-1];	//矩形上部分和左部分减去重复的
    for(int i=0;i<n;i++)for(int j=0;j<m;j++)
        arr[i][j]+=sum[i][j];				//进行原数组修改
    d=vector<vector<int>>(n+1,vector<int>(m+1,0))	//差分标记还原
}
```

## 整体二分

> 在$log_2n$复杂度内,对**有序**的序列进行操作

```c++
bool check(int x){}
int binary_check(int L,int R)
{
    //const double eps=1e-6;	//浮点数二分定义的精度
    //while(R-L>eps)		//浮点数二分的条件
    while(L<R)
    {
        int mid=(L+R)>>1;
        if(check(mid))R=mid;
        else L=mid+1;			//根据题目定义的区间更新方式  当l=mid时候,必须对mid进行偏右区间选择  防止r==l+1时候进入死循环
    }
    return L;		//返回下标
}
```

## 大整数运算

### 比较函数

```c++
bool cmp(vector<int> &A,vector<int> &B)		//小于返回真
{
    if(A.size()!=B.size())return A.size()<B.size();
    for(int i=A.size()-1;i>=0;i--)		//因为是倒序的
        if(A[i]!=B[i])return A[i]<B[i];
    return false;	
}	//只有!cmp(A,B)&&!cmp(B,A)才是相等
```

### 加法

```c++
//C=A+B  A>=0  B>=0
vector<int> add(vector<int> &A,vector<int> &B)	//返回的是倒序的大整数
{
    if(A.size()<(int)B.size())return add(B,A);	//保证A比B长
    vector<int> C;		//答案数组
    int t=0;
    for(int i=0;i<(int)A.size();i++)
    {
        t+=A[i];	//+A
        if(i<(int)B.size())t+=B[i];	//如果B位数还有 +B
        C.push_back(t%10);		//此处的10代表数组每个下标只存一位
        t/=10;					//根据需求还可以进行把多位数大数压进一个下标
    }
    if(t)C.push_back(t);	//还有剩的进位  加法保证进位只有一次
    return C;
}
```

### 减法

```c++
//C=A-B  A>=0  B>=0  A>=B
vector<int> sub(vector<int> &A,vector<int> &B)	//返回的是倒序的大整数
{
    vector<int> C;	//答案数组
    for(int i=0,t=0;i<(int)A.size();i++)
    {
        t=A[i]-t;	//A-借位
        if(i<(int)B.size())t-=B[i];		//如果B还有就 -B
        C.push_back((t+10)%10);		//防止减到负数	此处的10可以是更大
        if(t<0)t=1;
        else t=0;		//借位判断
    }
    while(C.size()>1&&C.back()==0)C.pop_back();	//去掉前缀0
    return C;
}
```

### 乘法

#### 大整数乘整数

```c++
//C=A*b  A>=0  b>0
vector<int> mul(vector<int> &A,int b)	//返回的是倒序的大整数
{
    vector<int> C;	//答案数组
    for(int i=0,t=0;i<(int)A.size()||t;i++)		//当A还有位数或者t还有进位
    {
        if(i<(int)A.size())t+=A[i]*b;
        C.push_back(t%10);		//此处的10可以是更大
        t/=10;
    }
    return C;
}
```

#### 大整数乘大整数

```c++
//C=A*B  A>=0  B>=0
vector<int> mul(vector<int> &A,vector<int> &B)	//返回的是倒序的大整数
{
    int s1=(int)A.size(),s2=(int)B.size();
    vector<int> C(s1+s2,0);	//答案数组且初始化为s1+s2长度
    for(int i=0;i<s1;i++)	
        for(int j=0;j<s2;j++){
            int sum=C[i+j]+A[i]*B[j];	//当前位数相乘加上进位
            C[i+j]=sum%10;		//答案当前位写入	此处的10可以是更大
            C[i+j+1]+=sum/10;	//进位
        }
    while(C.size()>1&&C.back()==0)C.pop_back();	//去除前缀0
    return C;
}
```

### 除法

#### 整数除以整数

```c++
//A/b=C......r  A>=0  b>0
vector<int>div(vector<int>&A,int b,int &r)	//返回的是倒序的大整数	余数存在r中
{
    vector<int> C;
    r=0;
    for(int i=(int)A.size()-1;i>=0;i--)	//从前往后除
    {
        r=r*10+A[i];	//模拟加0
        C.push_back(r/b);	//商
        r%=b;	//余数
    }
    reverse(C.begin(),C.end());	//把正序的反过来
    while(C.size()>1&&C.back()==0)C.pop_back();	//去除前缀0
    return C;
}
```

#### 大整数除以大整数

> 利用的是减法,把除数扩大等价于把减的次数扩大

```c++
//A/B=C  A>=0  B>0
vector<int> div(vector<int> A,vector<int> B)	//返回的是倒序的大整数且此处会修改AB
{
    int s1=A.size(),s2=B.size();
    int d=s1-s2;	//相差位数
    vector<int> C(d+1,0);	//答案数组  大小最大为长度相差+1
    reverse(B.begin(),B.end());
    for(int i=0;i<d;i++)B.push_back(0);		//在后面补0	即扩大倍数
    reverse(B.begin(),B.end());
    for(int i=0;i<=d;i++){
        while(!cmp(A,B)){		//当A大于等于B的时候	当A无法再进行减法的时候结束
            A=sub(A,B);		//减去
            C[d-i]++;		//对应商的位数计数++
        }
        B.erase(B.begin());	//缩小减数
    }
    while(C.size()>1&&C.back()==0)C.pop_back();		//去除前缀0
    return C;
}
```

### 取模

```c++
long long mod(vector<int> &A,int mod)
{
    int ans=0;
    for(int i=A.size()-1;i>=0;i--)
        ans=(ans*10+A[i])%mod;		//每步取模
    return ans;
}
```

### 输出

```c++
void out_num(vector<int> num)	//输出大整数
{
    for(int i=(int)num.size()-1;i>=0;i--)
        printf("%d",num[i]);	//根据压位的数可以改变输出的位数 如 %08d
    cout<<endl;
}
```

## 离散化

> 把无限空间有限的个体映射到有限的空间中去
>
> 可以是空间映射或者是大小映射

### 直接二分查找映射

```c++
vector<int> all;	//储存所有离散化的值  (下标)
const int N;	//离散化后的规模上限
vector<int> arr(N,0);	//储存数据的数组
sort(all.begin(),all.end());	//将所有值排序
all.erase(unique(all.begin(),all.end()),all.end());	//去除重复元素
unsigned int find_pos(int pos)
{
    return upper_bound(all.begin(),all.end(),pos)-all.begin();	//找到映射后对应的下标
}
// arr[find_pos(x)]=do something;		//全部操作改为映射操作
```

### 相对大小离散化

> 不改变相对大小和原来位置,对数据进行缩小

```c++
int n;	//数据量范围
struct Data{
    long long val;
    int pos;
    bool operator < (Data &B) const
    {
        return val<B.val;
    }
};
vector<Data> tmp(n);	//存放数据
vector<int> arr(n);		//原数组及 存放离散后的数据
void discretization()	//原数组下标应从1开始放标操作
{
    for(int i=1;i<arr.size();i++)		//进行数据位置和值的处理
    {
        tmp[i].val=arr[i];
        tmp[i].pos=i;
    }
    sort(tmp.begin()+1,tmp.end());
    tmp[0].val=1234567890;	//防止数据冲突
    int idx=0;
    for(int i=1;i<arr.size();i++){
        if(tmp[i-1].val==tmp[i].val)arr[tmp[i].pos]=idx;	//相同大小应该有相同的映射
        else arr[tmp[i].pos]=++idx;	//大小映射
    }
}
```

## 位运算

### lowbit

> 一个数二进制最右边的1与后面的0组成的数

```c++
inline int lowbit(int x){
    return x&-x;
}
```

### 求二进制数1的个数

> 直接通过公式: $x\&(x-1)$
>
> 因此可以得到一个数二进制1的个数

```c++
inline int num_of_1(int n){
    int cnt=0;
    while(n){
        cnt++;
        n=n&(n-1);
    }
    return cnt;
}


//也可以用库函数求
__builtin_popcount();	//返回二进制的1的个数 部分编译器不能用
```

### 预处理多个数的二进制1的个数

```c++
const int N;
int cnt[N];
for(int i=1;i<N;i++)cnt[i]=cnt[i>>1]+(i&1);
```

### 枚举集合的所有非空子集

```cpp
int S;	//某个集合
for(int sub=S;sub;sub=(sub-1)&S)
    //do something
```

### 暴力搜索

当题目说求某个合法子集的最大长度时

且集合总大小较小(<20)

