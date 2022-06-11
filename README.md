# LeetCode

## 二分搜索速记卡

* 704.二分查找

```java
public int search(int[] nums, int target) {
    int left=0,right=nums.length-1;
    while(left<=right){
        int mid=left+(right-left)/2;
        if(nums[mid]>target)
            right=mid-1;
        else if(nums[mid]<target)
            left=mid+1;
        else {
            return mid;
        }
    }
    return -1;
}
```

* 34.在排序数组中查找元素的第一个和最后一个位置

```java
public int leftfun(int[] nums, int target){
    int left=0,right=nums.length-1;
    while (left<=right){
        int mid=left+(right-left)/2;
        if(nums[mid]>=target)
            right=mid-1;
        else if(nums[mid]<target)
            left=mid+1;
    }
    if(left>=nums.length||nums[left]!=target)
        return -1;
    return left;
}
public int rightfun(int[] nums, int target){
    int left=0,right=nums.length-1;
    while (left<=right){
        int mid=left+(right-left)/2;
        if(nums[mid]>target)
            right=mid-1;
        else if(nums[mid]<=target)
            left=mid+1;
    }
    if(right<0||nums[right]!=target)
        return -1;
    return right;
}
public int[] searchRange(int[] nums, int target) {
    int x=leftfun(nums,target);
    int y=rightfun(nums,target);
    if(x>=0&&x<nums.length)
        return new int[]{x,y};
    return new int[]{-1,-1};
}
```

* 35.搜索插入位置

```java
public int searchInsert(int[] nums, int target) {
    int left=0,right=nums.length-1;
    while (left<=right){
        int mid=left+(right-left)/2;
        if(nums[mid]>target)
            right=mid-1;
        else if(nums[mid]<target)
            left=mid+1;
        else
            return mid;
    }
    return left;
}
```

* 392.判断子序列

```java
public boolean isSubsequence(String s, String t) {
    int s_len=s.length(),t_len=t.length();
    int i=0,j=0;
    while (i<s_len&&j<t_len){
        if(s.charAt(i)==t.charAt(j)){
            i++;
            j++;
        }else {
            j++;
        }
    }
    if(i==s_len)
        return true;
    else
        return false;
}
```

* 875.爱吃香蕉的珂珂

```java
public long hour(int[] piles, int mid){
    long h=0;
    for(int pile:piles){
        int m=pile/mid;
        int n=pile%mid;
        if(n==0)
            h+=m;
        else
            h+=m+1;
    }
    return h;
}
public int minEatingSpeed(int[] piles, int h) {
    int left=1,right=1;
    for(int pile :piles)
        right=Math.max(right,pile);
    while (left<=right){
        int mid=left+(right-left)/2;
        if(hour(piles,mid)>h)
            left=mid+1;
        else
            right=mid-1;
    }
    return right+1;
}
```

* 1011.在 D 天内送达包裹的能力

```java
public int day(int[] weights, int mid){
    int day=0;
    int sum=0;
    for(int weight:weights){
        if(weight+sum>=mid){
            if(weight+sum==mid){
                day++;
                sum=0;
            }else {
                day++;
                sum=weight;
            }
        }else {
            sum+=weight;
        }
    }
    if(sum==0)
        return day;
    else
        return day+1;
}
public int shipWithinDays(int[] weights, int days) {
    int left=1,right=1;
    for(int weight:weights){
        left=Math.max(left,weight);
        right+=weight;
    }
    while (left<=right){
        int mid=left+(right-left)/2;
        if(day(weights,mid)>days)
            left=mid+1;
        else
            right=mid-1;
    }
    return right+1;
}
```

* 354.俄罗斯套娃信封问题

```java
public int maxEnvelopes(int[][] envelopes) {
    int n=envelopes.length;
    Arrays.sort(envelopes, new Comparator<int[]>() {
        @Override
        public int compare(int[] o1, int[] o2) {
            return o1[0]==o2[0]?o2[1]-o1[1]:o1[0]-o2[0];
        }
    });
    int ygq[]=new int[n];
    int res=0;
    for(int i=0;i<n;i++){
        ygq[i]=1;
        for(int j=0;j<i;j++){
            if(envelopes[j][1]<envelopes[i][1]){
                ygq[i]=Math.max(ygq[j]+1,ygq[i]);
            }
        }
        res=Math.max(res,ygq[i]);
    }
    return res;
}
```
