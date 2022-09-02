# LeetCode

## 基础数据结构

### 二分搜索速记卡

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

### 滑动窗口速记卡

* 76.最小覆盖子串

```java
public String minWindow(String s, String t) {
    Map<Character,Integer> need=new HashMap<>();
    Map<Character,Integer> window=new HashMap<>();
    for(char c:t.toCharArray()){
        need.put(c,need.getOrDefault(c,0)+1);
    }
    int left=0,right=0;
    int valid=0;
    int start=0,len=Integer.MAX_VALUE;
    while (right<s.length()){
        char c=s.charAt(right);
        right++;
        if(need.containsKey(c)){
            window.put(c,window.getOrDefault(c,0)+1);
            if(need.get(c).equals(window.get(c)))
                valid++;
        }
        while (valid==need.size()){
            if(right-left<len){
                len=right-left;
                start=left;
            }
            char d=s.charAt(left);
            left++;
            if(need.containsKey(d)){
                if(need.get(d).equals(window.get(d)))
                    valid--;
                window.put(d,window.get(d)-1);
            }
        }
    }
    return len==Integer.MAX_VALUE?"":s.substring(start,start+len);
}
```

* 3.无重复字符的最长子串

```java
public int lengthOfLongestSubstring(String s) {
    Map<Character,Integer> window=new HashMap<>();
    int left=0,right=0,len=0;
    while (right<s.length()) {
        char c = s.charAt(right);
        right++;
        window.put(c, window.getOrDefault(c, 0) + 1);
        while (window.get(c) > 1) {
            char d = s.charAt(left);
            left++;
            window.put(d, window.get(d) - 1);
        }
        len=Math.max(len,right-left);
    }
    return len;
}
```

* 438.找到字符串中所有字母异位词

```java
public List<Integer> findAnagrams(String s, String p) {
    Map<Character,Integer> window=new HashMap<>();
    Map<Character,Integer> need=new HashMap<>();
    List<Integer> ygq=new ArrayList<>();
    for(char c:p.toCharArray())
        need.put(c,need.getOrDefault(c,0)+1);
    int left=0,right=0,valid=0;
    while (right<s.length()){
        char c=s.charAt(right);
        right++;
        if(need.containsKey(c)){
            window.put(c,window.getOrDefault(c,0)+1);
            if(need.get(c).equals(window.get(c)))
                valid++;
        }
        # notice！notice！notice！notice！notice！notice！notice！notice！notice！
        while (right-left>=p.length()){
            if(valid==need.size()){
                ygq.add(left);
            }
            char d=s.charAt(left);
            left++;
            if(need.containsKey(d)){
                if (need.get(d).equals(window.get(d)))
                    valid--;
                window.put(d,window.get(d)-1);
            }
        }
    }
    return ygq;
}
```

* 567.字符串的排列

```java
public boolean checkInclusion(String s1, String s2) {
    Map<Character,Integer> window=new HashMap<>();
    Map<Character,Integer> need=new HashMap<>();
    for(char c:s1.toCharArray())
        need.put(c,need.getOrDefault(c,0)+1);
    int left=0,right=0,valid=0;
    while (right<s2.length()){
        char c=s2.charAt(right);
        right++;
        if(need.containsKey(c)){
            window.put(c,window.getOrDefault(c,0)+1);
            if (need.get(c).equals(window.get(c)))
                valid++;
        }
        while (right-left>=s1.length()){
            if(valid==need.size())
                return true;
            char d=s2.charAt(left);
            left++;
            if(need.containsKey(d)){
                if(need.get(d).equals(window.get(d)))
                    valid--;
                window.put(d,window.get(d)-1);
            }
        }
    }
    return false;
}
```

* 239.滑动窗口最大值

```java
# 方案一：滑动窗口超时
public int maxfun(Queue<Integer> window){
    int max=window.peek();
    for(int tmp:window){
        max=Math.max(max,tmp);
    }
    return max;
}
public int[] maxSlidingWindow(int[] nums, int k) {
    Queue<Integer> window=new LinkedList<>();
    int[] ygq=new int[nums.length-k+1];
    int count=0;
    int left=0,right=0;
    while (right<nums.length){
        int c=nums[right];
        right++;
        window.add(c);
        while (right-left>=k){
            ygq[count++]=maxfun(window);
            int d=nums[left];
            left++;
            window.remove();
        }
    }
    return ygq;
}

# 方案二：单调队列（队列中存储的是索引、索引）
public int[] maxSlidingWindow(int[] nums, int k) {
    int n=nums.length;
    Deque<Integer> deque=new LinkedList<>();
    for(int i=0;i<k;i++){
        while (!deque.isEmpty()&&nums[i]>=nums[deque.peekLast()])
            deque.pollLast();
        deque.offerLast(i);
    }
    int[] ans=new int[n-k+1];
    ans[0]=nums[deque.peekFirst()];
    for(int i=k;i<n;i++){
        while (!deque.isEmpty()&&nums[i]>=nums[deque.peekLast()])
            deque.pollLast();
        deque.offerLast(i);
        while (deque.peekFirst()<=i-k){
            deque.pollFirst();
        }
        ans[i-k+1]=nums[deque.peekFirst()];
    }
    return ans;
}
```

### 其他双指针算法速记卡

* 26.删除有序数组中的重复项

```java
public int removeDuplicates(int[] nums) {
    if(nums.length==0)
        return 0;
    int slow=0,fast=0;
    while (fast<nums.length){
        if(nums[slow]!=nums[fast]){
            slow++;
            nums[slow]=nums[fast];
        }
        fast++;
    }
    return slow+1;
}
```

* 27.移除元素

```java
public int removeElement(int[] nums, int val) {
    if(nums.length==0)
        return 0;
    int slow=0,fast=0;
    while (fast<nums.length){
        if(nums[fast]!=val){
            nums[slow]=nums[fast];
            slow++;
        }
        fast++;
    }
    return slow;
}
```

* 283.移动零

```java
public void moveZeroes(int[] nums) {
    int slow=0,fast=0;
    while (fast<nums.length){
        if(nums[fast]!=0){
            nums[slow]=nums[fast];
            slow++;
        }
        fast++;
    }
    for(int i=slow;i<nums.length;i++){
        nums[i]=0;
    }
}
```

* 15.三数之和

```java
public List<List<Integer>> twoSum(int[] nums,int start,int target) {
    List<List<Integer>> ygq=new ArrayList<>();
    int left=start,right=nums.length-1;
    while (left<right){
        int n1=nums[left],n2=nums[right];
        int sum=nums[left]+nums[right];
        if(sum<target){
            while (left<right&&nums[left]==n1)left++;
        }
        else if(sum>target){
            while (left<right&&nums[right]==n2)right--;
        }
        else {
            List<Integer> tmp=new ArrayList<>();
            tmp.add(nums[left]);tmp.add(nums[right]);ygq.add(tmp);
            while (left<right&&nums[left]==n1)left++;
            while (left<right&&nums[right]==n2)right--;
        }
    }
    return ygq;
}
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> ygq=new ArrayList<>();
    Arrays.sort(nums);
    for(int i=0;i<nums.length;i++){
        List<List<Integer>> tmps=twoSum(nums,i+1, 0-nums[i]);
        for(List<Integer> tmp:tmps){
            tmp.add(nums[i]);
            ygq.add(tmp);
        }
        while (i<nums.length-1&&nums[i]==nums[i+1])i++;
    }
    return ygq;
}
```

* 18.四数之和

```java
public List<List<Integer>> twoSum(int[] nums, int start,int target) {
    int n=nums.length;
    int left=start,right=n-1;
    List<List<Integer>> ygq=new ArrayList<>();
    while (left<right){
        int sum=nums[left]+nums[right];
        int n1=nums[left],n2=nums[right];
        if(sum<target){
            while (left<right&&nums[left]==n1)left++;
        }else if(sum>target){
            while (left<right&&nums[right]==n2)right--;
        }else {
            List<Integer> tmp=new ArrayList<>();
            tmp.add(nums[left]);tmp.add(nums[right]);ygq.add(tmp);
            while (left<right&&nums[left]==n1)left++;
            while (left<right&&nums[right]==n2)right--;
        }
    }
    return ygq;
}

public List<List<Integer>> threeSum(int[] nums, int start,int target) {
    int n=nums.length;
    List<List<Integer>> ygq=new ArrayList<>();
    for(int i=start;i<n;i++){
        List<List<Integer>> tmps=twoSum(nums,i+1,target-nums[i]);
        for(List<Integer> tmp:tmps){
            tmp.add(nums[i]);
            ygq.add(tmp);
        }
        while (i<n-1&&nums[i]==nums[i+1])i++;
    }
    return ygq;
}

public List<List<Integer>> fourSum(int[] nums, int target) {
    int n=nums.length;
    Arrays.sort(nums);
    List<List<Integer>> ygq=new ArrayList<>();
    for(int i=0;i<n;i++){
        List<List<Integer>> tmps=threeSum(nums,i+1,target-nums[i]);
        for(List<Integer> tmp:tmps){
            tmp.add(nums[i]);
            ygq.add(tmp);
        }
        while (i<n-1&&nums[i]==nums[i+1])i++;
    }
    return ygq;
}
```

* 870.优势洗牌

```java
public int index(List<Integer> nums, int target){
    if(nums.size()==0)
        return -1;
    int left=0,right=nums.size()-1;
    while (left<=right){
        int mid=left+(right-left)/2;
        if(nums.get(mid)>target){
            right=mid-1;
        }else if(nums.get(mid)<target){
            left=mid+1;
        }else {
            while (mid<nums.size()&&nums.get(mid)==target)mid++;
            return mid;
        }
    }
    return left;
}

public int[] advantageCount(int[] nums1, int[] nums2) {
    Arrays.sort(nums1);
    int[] ygq=new int[nums1.length];
    int count=0;
    List<Integer> tmp=new ArrayList<>();
    for(int i:nums1)
        tmp.add(i);
    for(int i=0;i<nums2.length;i++){
        int index=index(tmp,nums2[i]);
        if(index>=tmp.size()){
            ygq[count++]=tmp.get(0);
            tmp.remove(0);
        }else {
            ygq[count++]=tmp.get(index);
            tmp.remove(index);
        }
    }
    return ygq;
}
```

* 42.接雨水

```java
public int trap(int[] height) {
    int left=0,right=height.length-1;
    int res=0;
    int l_max=height[left],r_max=height[right];
    while (left<right){
        l_max=Math.max(l_max,height[left]);
        r_max=Math.max(r_max,height[right]);
        if(l_max<r_max){
            res+=l_max-height[left];
            left++;
        }else {
            res+=r_max-height[right];
            right--;
        }
    }
    return res;
}
```

* 11.盛最多水的容器

```java
public int maxArea(int[] height) {
    int left=0,right=height.length-1;
    int res=0;
    while (left<right){
        res=Math.max(res,Math.min(height[right],height[left])*(right-left));
        if(height[left]<=height[right])
            left++;
        else
            right--;
    }
    return res;
}
```

### 链表双指针速记卡

* 2.两数相加

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode p1=l1,p2=l2;
    ListNode dummy=new ListNode(-1);
    ListNode p=dummy;
    int carry=0;
    while (p1!=null||p2!=null||carry!=0){
        int val=carry;
        if(p1!=null){
            val+=p1.val;
            p1=p1.next;
        }
        if(p2!=null){
            val+=p2.val;
            p2=p2.next;
        }
        carry=val/10;
        val=val%10;
        p.next=new ListNode(val);
        p=p.next;
    }
    return dummy.next;
}
```

* 19.删除链表的倒数第 N 个结点

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
     ListNode dummy=new ListNode(-1);
     dummy.next=head;
     ListNode slow=dummy;
     ListNode fast=dummy;
     for(int i=0;i<=n;i++){
         fast=fast.next;
     }
     while (fast!=null){
         fast=fast.next;
         slow=slow.next;
     }
     slow.next=slow.next.next;
     return dummy.next;
}
```

* 21.合并两个有序链表

```java
public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
    ListNode dummy=new ListNode(-1);
    ListNode p=dummy;
    while (list1!=null||list2!=null){
        if(list1==null){
            p.next=list2;
            break;
        }else if(list2==null){
            p.next=list1;
            break;
        }else {
            if(list1.val<list2.val){
                p.next=list1;
                p=p.next;
                list1=list1.next;
            }else {
                p.next=list2;
                p=p.next;
                list2=list2.next;
            }
        }

    }
    return dummy.next;
}
```

* 23.合并K个升序链表

```java
public ListNode mergeKLists(ListNode[] lists) {
     if(lists.length==0)
         return null;
     ListNode dummy=new ListNode(-1);
     ListNode p=dummy;
    PriorityQueue<ListNode> pq=new PriorityQueue<>(lists.length,(a,b)->(a.val-b.val));
    for(ListNode list:lists){
        if(list!=null)
            pq.add(list);
    }
    while (!pq.isEmpty()){
        ListNode node=pq.poll();
        p.next=node;
        p=p.next;
        if(node.next!=null)
            pq.add(node.next);
    }
    return dummy.next;
}
```

* 141.环形链表

```java
public boolean hasCycle(ListNode head) {
     if(head==null)
         return false;
    ListNode slow=head,fast=head;
    while (fast.next!=null&&fast.next.next!=null){
        fast=fast.next.next;
        slow=slow.next;
        if(fast==slow)
            return true;
    }
    return false;
}
```

* 142.环形链表II

```java
public ListNode detectCycle(ListNode head) {
    ListNode dummy=new ListNode(-1);
    ListNode p=dummy;
    ListNode slow=head,fast=head;
    while (fast!=null&&fast.next!=null&&fast.next.next!=null){
        slow=slow.next;
        fast=fast.next.next;
        if(slow==fast){
            fast=head;
            while (slow!=fast){
                slow=slow.next;
                fast=fast.next;
            }
            return slow;
        }
    }
    return null;
}
```

* 160.相交链表

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode p1=headA,p2=headB;
    while (p1!=p2&&p1!=null){
        while (p2!=null){
            p2=p2.next;
            if(p1==p2){
                return p1;
            }
        }
        p2=headB;
        p1=p1.next;
    }
    return p1;
}
```

* 876.链表的中间结点

```java
public ListNode middleNode(ListNode head) {
     if(head==null)
         return null;
     ListNode slow=head,fast=head;
     while (fast.next!=null&&fast.next.next!=null){
         fast=fast.next.next;
         slow=slow.next;
     }
     if(fast.next==null)
         return slow;
     else
         return slow.next;
}
```

* 25.K个一组翻转链表（*）

```java
public ListNode reverseKGroup(ListNode head, int k) {
    if (head == null) return null;
    // 区间 [a, b) 包含 k 个待反转元素
    ListNode a, b;
    a = b = head;
    for (int i = 0; i < k; i++) {
        // 不足 k 个，不需要反转，base case
        if (b == null) return head;
        b = b.next;
    }
    // 反转前 k 个元素
    ListNode newHead = reverse(a, b);
    // 递归反转后续链表并连接起来
    a.next = reverseKGroup(b, k);
    return newHead;
}

/* 反转区间 [a, b) 的元素，注意是左闭右开 */
ListNode reverse(ListNode a, ListNode b) {
    ListNode pre, cur, nxt;
    pre = null;
    cur = a;
    nxt = a;
    // while 终止的条件改一下就行了
    while (cur != b) {
        nxt = cur.next;
        cur.next = pre;
        pre = cur;
        cur = nxt;
    }
    // 返回反转后的头结点
    return pre;
}
```

* 83.删除排序链表中的重复元素

```java
public ListNode deleteDuplicates(ListNode head) {
    if(head==null)
        return head;
    ListNode slow=head,fast=head.next;
    while (fast!=null){
        if(fast.val==slow.val){
            fast=fast.next;
        }else {
            slow.next=fast;
            slow=slow.next;
            fast=fast.next;
        }
    }
    slow.next=null;
    return head;
}
```

* 92.反转链表II

```java
 public ListNode reverse(ListNode slow,ListNode fast){
     ListNode pre=null;
     ListNode cur=slow;
     ListNode next=slow;
     while (cur!=fast){
         next=cur.next;
         cur.next=pre;
         pre=cur;
         cur=next;
     }
     return pre;
 }
public ListNode reverseBetween(ListNode head, int left, int right) {
    if(head==null||left==right)
        return head;
    ListNode dummy=new ListNode(-1);
    dummy.next=head;
    ListNode slow=head,fast=head;
    ListNode tmp=dummy;
    for(int i=0;i<right;i++){
        if(i<left-1){
            tmp=slow;
            slow=slow.next;
        }
        fast=fast.next;
    }
    tmp.next=reverse(slow,fast);
    slow.next=fast;
    return dummy.next;
}
```

* 234.回文链表

```java
public boolean isPalindrome(ListNode head) {
    ListNode dummy=new ListNode(-1);
    ListNode p2=head;
    while (p2!=null){
        ListNode tmp=dummy.next;
        dummy.next=new ListNode(p2.val);
        dummy.next.next=tmp;
        p2=p2.next;
    }
    ListNode p1=dummy.next;
    while (head!=null){
        if(head.val!=p1.val)
            return false;
        p1=p1.next;
        head=head.next;
    }
    return true;
}
```

### 前缀和技巧速记卡

* 303.区域和检索 - 数组不可变

```java
private int ygq[];
public NumArray(int[] nums) {
    ygq=new int[nums.length+1];
    ygq[0]=0;
    for(int i=1;i<=nums.length;i++){
        ygq[i]=ygq[i-1]+nums[i-1];
    }
}
public int sumRange(int left, int right) {
    return ygq[right+1]-ygq[left];
}
```

* 304.二维区域和检索 - 矩阵不可变

```java
private int[][] ygq;
public NumMatrix(int[][] matrix) {
    ygq=new int[matrix.length+1][matrix[0].length+1];
    for(int i=1;i<=matrix.length;i++){
        for(int j=1;j<=matrix[0].length;j++){
            ygq[i][j]=ygq[i-1][j]+ygq[i][j-1]-ygq[i-1][j-1]+matrix[i-1][j-1];
        }
    }
}

public int sumRegion(int row1, int col1, int row2, int col2) {
    return ygq[row2+1][col2+1]-ygq[row2+1][col1]-ygq[row1][col2+1]+ygq[row1][col1];
}
```

* 560.和为K的子数组

```java
public int subarraySum(int[] nums, int k) {
    int ygq[]=new int[nums.length+1];
    ygq[0]=0;
    for(int i=1;i<nums.length+1;i++){
        ygq[i]=ygq[i-1]+nums[i-1];
    }
    int count=0;
    for(int i=0;i<nums.length;i++){
        for(int j=i+1;j<nums.length+1;j++){
            if(ygq[j]-ygq[i]==k)
                count++;
        }
    }
    return count;
}
```

### 查分数组速记卡

* 1109.航班预订统计

```java
public int[] corpFlightBookings(int[][] bookings, int n) {
    Difference df=new Difference(new int[n]);
    for(int[] booking:bookings){
        int i=booking[0]-1;
        int j=booking[1]-1;
        int val=booking[2];
        df.increment(i,j,val);
    }
    return df.result();
}
class Difference{
    private int[] df;
    public Difference(int[] nums){
        df=new int[nums.length];
        df[0]=nums[0];
        for(int i=1;i<nums.length;i++){
            df[i]=nums[i]-nums[i-1];
        }
    }

    public void increment(int i,int j,int val){
        df[i]+=val;
        if(j+1<df.length)
            df[j+1]-=val;
    }

    public int[] result(){
        int[] ygq=new int[df.length];
        ygq[0]=df[0];
        for(int i=1;i<df.length;i++)
            ygq[i]=ygq[i-1]+df[i];
        return ygq;
    }
}
```

* 1094.拼车

```java
public boolean carPooling(int[][] trips, int capacity) {
    int[] nums=new int[1001];
    Difference df=new Difference(nums);
    for(int[] trip:trips){
        int i=trip[1];
        int j=trip[2]-1;
        int val=trip[0];
        df.increment(i,j,val);
    }
    int[] ygq=df.result();
    for(int i=0;i<ygq.length;i++){
        if(ygq[i]>capacity)
            return false;
    }
    return true;
}
class Difference{
     private int[] df;
     public Difference(int[] nums){
         df=new int[nums.length];
         df[0]=nums[0];
         for(int i=1;i<nums.length;i++)
             df[i]=nums[i]-nums[i-1];
     }
     public void increment(int i,int j,int val){
         df[i]+=val;
         if(j+1<df.length)
             df[j+1]-=val;
     }
     public int[] result(){
         int ygq[]=new int[df.length];
         ygq[0]=df[0];
         for(int i=1;i<ygq.length;i++){
             ygq[i]=ygq[i-1]+df[i];
         }
         return ygq;
     }
}
```

* 1.两数之和（#）

```java
public int[] twoSum(int[] nums, int target) {
    Map<Integer, Integer> hashtable = new HashMap<Integer, Integer>();
    for (int i = 0; i < nums.length; ++i) {
        if (hashtable.containsKey(target - nums[i])) {
            return new int[]{hashtable.get(target - nums[i]), i};
        }
        hashtable.put(nums[i], i);
    }
    return new int[0];
}
```

* 88.合并两个有序数组（#）

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int p1 = 0, p2 = 0;
    int[] sorted = new int[m + n];
    int cur;
    while (p1 < m || p2 < n) {
        if (p1 == m) {
            cur = nums2[p2++];
        } else if (p2 == n) {
            cur = nums1[p1++];
        } else if (nums1[p1] < nums2[p2]) {
            cur = nums1[p1++];
        } else {
            cur = nums2[p2++];
        }
        sorted[p1 + p2 - 1] = cur;
    }
    for (int i = 0; i != m + n; ++i) {
        nums1[i] = sorted[i];
    }
}
```

### 队列/栈算法速记卡

* 232.用栈实现队列

```java
private Stack<Integer> s1,s2;

public MyQueue() {
    s1=new Stack<>();
    s2=new Stack<>();
}

public void push(int x) {
    s1.push(x);
}

public int pop() {
    peek();
    return s2.pop();
}

public int peek() {
    if(s2.isEmpty()){
        while (!s1.isEmpty())
            s2.push(s1.pop());
    }
    return s2.peek();
}

public boolean empty() {
    return s1.isEmpty()&&s2.isEmpty();
}
```

* 225.用队列实现栈

```java
private Queue<Integer> q=new LinkedList<>();
int top_elem=0;
public MyStack() {

}

public void push(int x) {
    q.offer(x);
    top_elem=x;
}

public int pop() {
    int size=q.size();
    while (size>2){
        q.offer(q.poll());
        size--;
    }
    top_elem=q.peek();
    q.offer(q.poll());
    return q.poll();
}

public int top() {
    return top_elem;
}

public boolean empty() {
    return q.isEmpty();
}
```

* 32.最长有效括号

```java
public int longestValidParentheses(String s) {
    Stack<Integer> stack=new Stack<>();
    int[] dp=new int[s.length()];
    for(int i=0;i<s.length();i++){
        if(s.charAt(i)=='('){
            stack.push(i);
            dp[i+1]=0;
        }else {
            if(!stack.isEmpty()){
                int left=stack.pop();
                int len=1+i-left+dp[left];
                dp[i+1]=len;
            }else {
                dp[i+1]=0;
            }
        }
    }
    int res=0;
    for(int i=0;i<dp.length;i++)
        res=Math.max(res,dp[i]);
    return res;
}
```

* 1541.平衡括号字符串的最少插入次数

```java
public int minInsertions(String s) {
    int res=0,need=0;
    for(int i=0;i<s.length();i++){
        if(s.charAt(i)=='('){
            need+=2;
            if(need%2==1){
                res++;
                need--;
            }
        }else {
            need--;
            if(need==-1){
                res++;
                need=1;
            }
        }
    }
    return res+need;
}
```

* 921.使括号有效的最少添加

```java
public int minAddToMakeValid(String s) {
    int res=0,need=0;
    Stack<Character> stack=new Stack<>();
    for(int i=0;i<s.length();i++){
        if(s.charAt(i)=='('){
            need++;
            stack.push(s.charAt(i));
        }else {
            if(!stack.isEmpty()){
                need--;
                stack.pop();
            }else {
                res++;
            }
        }
    }
    return res+need;
}
```

* 20.有效的括号

```java
public boolean isValid(String s) {
    Stack<Character> stack=new Stack<>();
    for(int i=0;i<s.length();i++){
        char c=s.charAt(i);
        if(c=='('||c=='['||c=='{'){
            stack.push(c);
        }else {
            if(!stack.isEmpty()){
                char d=stack.peek();
                if((d=='('&&c==')')||(d=='['&&c==']')||(d=='{'&&c=='}'))
                    stack.pop();
                else
                    return false;
            }else {
                return false;
            }
        }
    }
    if(stack.isEmpty())
        return true;
    else
        return false;
}
```

### 二叉堆算法速记卡

* 23.合并K个升序链表

```java
public ListNode mergeKLists(ListNode[] lists) {
     if(lists.length==0)
         return null;
     ListNode dummy=new ListNode(-1);
     ListNode p=dummy;
    PriorityQueue<ListNode> pq=new PriorityQueue<>(lists.length,(a,b)->(a.val-b.val));
    for(ListNode list:lists){
        if(list!=null)
            pq.add(list);
    }
    while (!pq.isEmpty()){
        ListNode node=pq.poll();
        p.next=node;
        p=p.next;
        if(node.next!=null)
            pq.add(node.next);
    }
    return dummy.next;
}
```

* 215.数组中的第K个最大元素

```java
public int findKthLargest(int[] nums, int k) {
    PriorityQueue<Integer> pq=new PriorityQueue<>();
    for(int num:nums){
        pq.offer(num);
        if(pq.size()>k)
            pq.poll();
    }
    return pq.peek();
}
```

* 295.数据流的中位数

```java
PriorityQueue<Integer> large;
PriorityQueue<Integer> small;
public MedianFinder() {
    large=new PriorityQueue<>((a,b)->(b-a));
    small=new PriorityQueue<>();
}

public void addNum(int num) {
    if(small.size()>=large.size()){
        small.add(num);
        large.offer(small.poll());
    }else {
        large.add(num);
        small.offer(large.poll());
    }
}

public double findMedian() {
    if(small.size()>large.size())
        return small.peek();
    else if(small.size()<large.size())
        return large.peek();
    else
        return (small.peek()+large.peek())/2.0;
}
```

## 进阶数据结构

### 二叉树算法速记卡

* 104.二叉树的最大深度

```java
public int maxDepth(TreeNode root) {
    if(root==null)
        return 0;
    int len1=maxDepth(root.left);
    int len2=maxDepth(root.right);
    return 1+Math.max(len1,len2);
}
```

* 105.从前序与中序遍历序列构造二叉树

```java
private Map<Integer,Integer> index;
public TreeNode build(int[] preorder, int[] inorder,int pre_left,int pre_right,int in_left,int in_right){
    if(pre_left>pre_right)
        return null;
    int pre_root=pre_left;
    int in_root=index.get(preorder[pre_root]);
    TreeNode root=new TreeNode(preorder[pre_root]);
    int left_size=in_root-in_left;
    root.left=build(preorder,inorder,pre_left+1,pre_left+left_size,in_left,in_root-1);
    root.right=build(preorder,inorder,pre_left+left_size+1,pre_right,in_root+1,in_right);
    return root;
}
public TreeNode buildTree(int[] preorder, int[] inorder) {
    int n=preorder.length;
    index=new HashMap<>();
    for(int i=0;i<n;i++)
        index.put(inorder[i],i);
    return build(preorder,inorder,0,n-1,0,n-1);
}
```

* 106.从中序与后序遍历序列构造二叉树

```java
private Map<Integer,Integer> index;
public TreeNode build(int[] inorder, int[] postorder,int in_left,int in_right,int p_left,int p_right){
    if(in_left>in_right)
        return null;
    int p_root=p_right;
    int in_root=index.get(postorder[p_root]);
    int left_size=in_root-in_left;
    TreeNode root=new TreeNode(inorder[in_root]);
    root.left=build(inorder,postorder,in_left,in_root-1,p_left,p_left+left_size-1);
    root.right=build(inorder,postorder,in_root+1,in_right,p_left+left_size,p_root-1);
    return root;
}
public TreeNode buildTree(int[] inorder, int[] postorder) {
    index=new HashMap<>();
    int n=inorder.length;
    for(int i=0;i<n;i++)
        index.put(inorder[i],i);
    return build(inorder,postorder,0,n-1,0,n-1);
}
```

* 102.二叉树的层序遍历

```java
public List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> ygq=new LinkedList<>();
    if(root==null)
        return ygq;
    Queue<TreeNode> queue=new LinkedList<>();
    queue.offer(root);
    while (!queue.isEmpty()){
        int size=queue.size();
        List<Integer> tmp=new ArrayList<>();
        for(int i=0;i<size;i++){
            TreeNode node=queue.poll();
            tmp.add(node.val);
            if(node.left!=null)
                queue.offer(node.left);
            if(node.right!=null)
                queue.offer(node.right);
        }
        ygq.add(tmp);
    }
    return ygq;
}
```

* 111.二叉树的最小深度

```java
public int minDepth(TreeNode root) {
    if(root==null)
        return 0;
    if(root.left!=null&&root.right!=null){
        int left=minDepth(root.left);
        int right=minDepth(root.right);
        return Math.min(left,right)+1;
    }else if(root.left!=null&&root.right==null){
        return minDepth(root.left)+1;
    }else if(root.left==null&&root.right!=null){
        return minDepth(root.right)+1;
    }else {
        return 1;
    }
}
```

* 最大二叉树

```java
public TreeNode c_Tree(int[] nums,int left,int right) {
    if(left>right)
        return null;
    int n_max=nums[left];
    int count=left;
    for(int i=left;i<=right;i++){
        if(nums[i]>=n_max){
            n_max=nums[i];
            count=i;
        }
    }
    TreeNode root=new TreeNode(n_max);
    root.left=c_Tree(nums,left,count-1);
    root.right=c_Tree(nums,count+1,right);
    return root;
}
public TreeNode constructMaximumBinaryTree(int[] nums) {
    int n=nums.length;
    if(n==0)
        return null;
    return c_Tree(nums,0,n-1);
}
```

* 114.二叉树展开为链表

```java
public void flatten(TreeNode root) {
    if(root==null)
        return;
    flatten(root.left);
    flatten(root.right);
    TreeNode left=root.left;
    TreeNode right=root.right;
    root.left=null;
    root.right=left;
    TreeNode p=root;
    while (p.right!=null)
        p=p.right;
    p.right=right;
}
```

* 116.填充每个节点的下一个右侧节点指针

```java
public Node connect(Node root) {
    if(root==null)
        return null;
    connettwo(root.left,root.right);
    return root;
}
public void connettwo(Node node1,Node node2){
    if(node1==null||node2==null)
        return;
    node1.next=node2;
    connettwo(node1.left,node1.right);
    connettwo(node2.left,node2.right);
    connettwo(node1.right,node2.left);
}
```

* 226.翻转二叉树

```java
public TreeNode invertTree(TreeNode root) {
    if(root==null)
        return null;
    TreeNode tmp=root.left;
    root.left=root.right;
    root.right=tmp;
    invertTree(root.left);
    invertTree(root.right);
    return root;
}
```

* 297.二叉树的序列化与反序列化

```java
// Encodes a tree to a single string.
public String serialize(TreeNode root) {
    return reserialize(root,"");
}

// Decodes your encoded data to tree.
public TreeNode deserialize(String data) {
    String[] datastr=data.split(",");
    List<String> dataList=new ArrayList<>(Arrays.asList(datastr));
    return redeserialize(dataList);
}

public String reserialize(TreeNode root,String str){
    if(root==null){
        str+="None,";
    }else{
        str+=String.valueOf(root.val)+",";
        str=reserialize(root.left,str);
        str=reserialize(root.right,str);
    }
    return str;
}

public TreeNode redeserialize(List<String> dataList){
    if(dataList.get(0).equals("None")){
        dataList.remove(0);
        return null;
    }
    TreeNode root=new TreeNode(Integer.valueOf(dataList.get(0)));
    dataList.remove(0);
    root.left=redeserialize(dataList);
    root.right=redeserialize(dataList);
    return root;
}
```

* 652.寻找重复的子树

```java
Map<String,Integer> map;
List<TreeNode> node;
public List<TreeNode> findDuplicateSubtrees(TreeNode root) {
    map=new HashMap<>();
    node=new ArrayList<>();
    collect(root);
    return node;
}
public String collect(TreeNode root){
    if(root==null)
        return "#";
    String str=String.valueOf(root.val)+","+collect(root.left)+","+collect(root.right);
    map.put(str,map.getOrDefault(str,0)+1);
    if(map.get(str)==2)
        node.add(root);
    return str;
}
```

* 222.完全二叉树的节点个数

```java
public int countNodes(TreeNode root) {
    if(root==null)
        return 0;
    int left=countNodes(root.left);
    int right=countNodes(root.right);
    return left+right+1;
}
```

### 二叉搜索树速记卡

* 700.二叉搜索树中的搜索

```java
public TreeNode searchBST(TreeNode root, int val) {
    if(root==null)
        return null;
    if(root.val>val)
        return searchBST(root.left,val);
    if(root.val<val)
        return searchBST(root.right,val);
    return root;
}
```

* 701.二叉搜索树中的插入操作

```java
public TreeNode insertIntoBST(TreeNode root, int val) {
    if(root==null)
        return new TreeNode(val);
    inserT(root,val);
    return root;
}
public void inserT(TreeNode root, int val){
    if(root.val<val){
        if(root.right==null){
            root.right=new TreeNode(val);
            return;
        }else {
            inserT(root.right,val);
        }
    }
    if(root.val>val){
        if(root.left==null){
            root.left=new TreeNode(val);
            return;
        }else {
            inserT(root.left,val);
        }
    }
    return;
}
```

* 450.删除二叉搜索树中的节点

```java
public TreeNode deleteNode(TreeNode root, int key) {
    if(root==null)
        return null;
    if(root.val==key){
        if(root.left==null)
            return root.right;
        if(root.right==null)
            return root.left;
        TreeNode tmp=find(root.right);
        deleteNode(root,tmp.val);
        tmp.left=root.left;
        tmp.right=root.right;
        root=tmp;
    }else if(root.val>key){
        root.left=deleteNode(root.left,key);
    }else {
        root.right=deleteNode(root.right,key);
    }
    return root;
}
public TreeNode find(TreeNode root){
    while (root.left!=null)
        root=root.left;
    return root;
}
```

* 98.验证二叉搜索树

```java
public boolean isValidBST(TreeNode root) {
    return ValidBST(root,null,null);
}
public boolean ValidBST(TreeNode root,TreeNode min,TreeNode max) {
    if(root==null)
        return true;
    if(min!=null&&root.val<=min.val)return false;
    if(max!=null&&root.val>=max.val)return false;
    return ValidBST(root.left,min,root)&&ValidBST(root.right,root,max);
}
```

* 230.二叉搜索树中第K小的元素

```java
private int count=0;
private int ans=0;
public int kthSmallest(TreeNode root, int k) {
    Kth(root,k);
    return ans;
}
public void Kth(TreeNode root,int k){
    if(root==null)
        return;
    Kth(root.left,k);
    count++;
    if(count==k){
        ans=root.val;
        return;
    }
    Kth(root.right,k);
}
```

* 96.不同的二叉搜索树

```java
public int numTrees(int n) {
    int[] dp=new int[n+1];
    dp[0]=1;
    dp[1]=1;
    for(int i=2;i<n+1;i++){
        for(int j=1;j<=i;j++){
            dp[i]+=dp[j-1]*dp[i-j];
        }
    }
    return dp[n];
}
```

* 95.不同的二叉搜索树II

```java
public List<TreeNode> generateTrees(int n) {
    if(n==0)
        return new LinkedList<>();
    return build(1,n);
}
public List<TreeNode> build(int low,int high){
    List<TreeNode> res=new LinkedList<>();
    if(low>high){
        res.add(null);
        return res;
    }
    for(int i=low;i<=high;i++){
        List<TreeNode> left=build(low,i-1);
        List<TreeNode> right=build(i+1,high);
        for(TreeNode le:left){
            for(TreeNode ri:right){
                TreeNode root=new TreeNode(i);
                root.left=le;
                root.right=ri;
                res.add(root);
            }
        }
    }
    return res;
}
```

* 538.把二叉搜索树转换为累加树

```java
int sum = 0;
public TreeNode convertBST(TreeNode root) {
    if(root==null)
        return null;
    convert(root);
    return root;
}
public void convert(TreeNode root){
    if(root==null)
        return ;
    convert(root.right);
    sum+=root.val;
    root.val=sum;
    convert(root.left);
}
```

* 1373.二叉搜索子树的最大键值和

```java
int ans=0;
public int maxSumBST(TreeNode root) {
    isBST(root);
    return ans;
}
public int[] isBST(TreeNode root){
    if(root==null)
        return new int[]{1,Integer.MAX_VALUE,Integer.MIN_VALUE,0};
    int[] left=isBST(root.left);
    int[] right=isBST(root.right);
    int[] res=new int[4];
    if(left[0]==1&&right[0]==1&&root.val>left[2]&&root.val<right[1]){
        res[0]=1;
        res[1]=Math.min(root.val,left[1]);
        res[2]=Math.max(root.val,right[2]);
        res[3]=left[3]+right[3]+root.val;
        ans=Math.max(ans,res[3]);
    }
    return res;
}
```

### 图论算法速记卡

* 797.所有可能的路径

```java
List<List<Integer>> ygq=new ArrayList<List<Integer>>();
Stack<Integer> stack=new Stack<>();
public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
    stack.push(0);
    dfs(graph,stack,0,graph.length-1);
    return ygq;
}
public void dfs(int[][] graph,Stack<Integer> stack,int x,int n){
    if(x==n){
        ygq.add(new ArrayList<>(stack));
        return;
    }
    for(int tmp:graph[x]){
        stack.add(tmp);
        dfs(graph,stack,tmp,n);
        stack.pop();
    }
}
```

* 785.判断二分图

```java
private boolean[] color;
private boolean[] visited;
private boolean ok=true;
public boolean isBipartite(int[][] graph) {
    int n=graph.length;
    color=new boolean[n];
    visited=new boolean[n];
    for(int i=0;i<n;i++){
        if(!visited[i]){
            traverse(graph,i);
        }
    }
    return ok;
}
public void traverse(int[][] graph,int v){
    if(!ok)
        return;
    visited[v]=true;
    for(int i:graph[v]){
        if(!visited[i]){
            color[i]=!color[v];
            traverse(graph,i);
        }else {
            if(color[i]==color[v])
                ok=false;
        }
    }
}
```

* 886.可能的二分法

```java
private boolean[] color;
private boolean[] visited;
private boolean ok=true;
public boolean possibleBipartition(int n, int[][] dislikes) {
    color=new boolean[n+1];
    visited=new boolean[n+1];
    List<Integer>[] graph=build(dislikes,n+1);
    for(int i=1;i<=n;i++){
        if(!visited[i]){
            traserve(graph,i);
        }
    }
    return ok;
}
public List<Integer>[] build(int[][] dislikes,int n){
    List<Integer>[] graph=new LinkedList[n];
    for(int i=1;i<n;i++){
        graph[i]=new LinkedList<>();
    }
    for(int[] e:dislikes){
        int v=e[0];
        int w=e[1];
        graph[v].add(w);
        graph[w].add(v);
    }
    return graph;
}
public void traserve(List<Integer>[] graph,int v){
    if(!ok)
        return;
    visited[v]=true;
    for(int i:graph[v]){
        if(!visited[i]){
            color[i]=!color[v];
            traserve(graph,i);
        }else {
            if(color[v]==color[i])
                ok=false;
        }
    }
}
```

* 207.课程表

```java
private boolean[] onpath;
private boolean[] visited;
private boolean ok=false;
public boolean canFinish(int numCourses, int[][] prerequisites) {
    onpath=new boolean[numCourses];
    visited=new boolean[numCourses];
    List<Integer>[] graph=build(prerequisites,numCourses);
    for(int i=0;i<numCourses;i++){
        traverse(graph,i);
    }
    return !ok;
}
public void traverse(List<Integer>[] graph,int v){
    if(onpath[v])
        ok=true;
    if(visited[v]||ok)
        return;
    visited[v]=true;
    onpath[v]=true;
    for(int i:graph[v]){
        traverse(graph,i);
    }
    onpath[v]=false;
}
public List<Integer>[] build(int[][] prerequisites,int numCourses){
    List<Integer>[] graph=new LinkedList[numCourses];
    for(int i=0;i<numCourses;i++)
        graph[i]=new LinkedList<>();
    for(int[] tmp:prerequisites){
        int v=tmp[0];
        int w=tmp[1];
        graph[w].add(v);
    }
    return graph;
}
```

* 210.课程表II

```java
List<Integer> ygq=new ArrayList<>();
private boolean[] visited;
private boolean[] onpath;
private boolean cycle=false;
public int[] findOrder(int numCourses, int[][] prerequisites) {
    visited=new boolean[numCourses];
    onpath=new boolean[numCourses];
    List<Integer>[] graph=build(prerequisites,numCourses);
    for(int i=0;i<numCourses;i++){
        traverse(graph,i);
    }
    if(cycle)
        return new int[]{};
    int[] res=new int[numCourses];
    for(int i=0;i<numCourses;i++){
        res[numCourses-i-1]=ygq.get(i);
    }
    return res;
}
public void traverse(List<Integer>[] graph,int v){
    if(onpath[v])
        cycle=true;
    if(visited[v]||cycle)
        return;
    visited[v]=true;
    onpath[v]=true;
    for(int tmp:graph[v]){
        traverse(graph,tmp);
    }
    ygq.add(v);
    onpath[v]=false;
}
public List<Integer>[] build(int[][] prerequisites,int numCourses){
    List<Integer>[] graph=new LinkedList[numCourses];
    for(int i=0;i<numCourses;i++)
        graph[i]=new LinkedList<>();
    for(int[] tmp:prerequisites){
        int v=tmp[0];
        int w=tmp[1];
        graph[w].add(v);
    }
    return graph;
}
```

* 130.被围绕的区域

```java
public class Solution {
    public void solve(char[][] board) {
        if(board.length==0)
            return;
        int m=board.length;
        int n=board[0].length;
        UF uf=new UF(m*n+1);
        int dummy=m*n;
        for(int i=0;i<m;i++){
            if(board[i][0]=='O')
                uf.union(i*n+0,dummy);
            if(board[i][n-1]=='O')
                uf.union(i*n+n-1,dummy);
        }
        for(int j=0;j<n;j++){
            if(board[0][j]=='O')
                uf.union(n*0+j,dummy);
            if(board[m-1][j]=='O')
                uf.union(n*(m-1)+j,dummy);
        }
        int[][] d=new int[][]{{1, 0}, {0, 1}, {0, -1}, {-1, 0}};
        for(int i=1;i<m-1;i++){
            for(int j=1;j<n-1;j++){
                if(board[i][j]=='O'){
                    for(int k=0;k<4;k++){
                        int x = i + d[k][0];
                        int y = j + d[k][1];
                        if (board[x][y] == 'O'){
                            uf.union(x*n+y,i*n+j);
                        }
                    }
                }
            }
        }
        for(int i=1;i<m-1;i++){
            for(int j=1;j<n-1;j++){
                if(!uf.connected(dummy,i*n+j)){
                    board[i][j]='X';
                }
            }
        }
    }
}
class UF{
    private int count;
    private int[] partent;
    private int[] size;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<n;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]>size[rootq]){
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }else {
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (partent[x]!=x){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
}
```

* 990.等式方程的可满足性

```java
public class Solution {
    public boolean equationsPossible(String[] equations) {
        UF uf=new UF(26);
        for(String c:equations){
            if(c.charAt(1)=='='){
                uf.union(c.charAt(0)-'a',c.charAt(3)-'a');
            }
        }
        for(String c:equations){
            if(c.charAt(1)=='!'){
                if(uf.connected(c.charAt(0)-'a',c.charAt(3)-'a'))
                    return false;
            }
        }
        return true;
    }
}
class UF{
    private int count;
    private int[] partent;
    private int[] size;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<n;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]>size[rootq]){
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }else {
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (partent[x]!=x){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
}
```

* 1584.连接所有点的最小费用

```java
public int minCostConnectPoints(int[][] points) {
    int n=points.length;
    List<int[]> edges=new ArrayList<>();
    for(int i=0;i<n;i++){
        for(int j=i+1;j<n;j++){
            int xi=points[i][0],yi=points[i][1];
            int xj=points[j][0],yj=points[j][1];
            edges.add(new int[]{i,j,Math.abs(xi-xj)+Math.abs(yi-yj)});
        }
    }
    Collections.sort(edges,(a,b)->{return a[2]-b[2];});
    int mst=0;
    UF uf=new UF(n);
    for(int[] edge:edges){
        int u=edge[0];
        int v=edge[1];
        int w=edge[2];
        if(uf.connected(u,v))
            continue;
        uf.union(u,v);
        mst+=w;
    }
    return mst;
}
class UF{
    private int partent[];
    private int size[];
    private int count;
    public UF(int n){
        partent=new int[n];
        size=new int[n];
        count=n;
        for(int i=0;i<n;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]<size[rootq]){
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }else {
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (partent[x]!=x){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
    public int count(){
        return count;
    }
}
```

* 743.网络延迟时间

```java
public int networkDelayTime(int[][] times, int n, int k) {
    List<int[]>[] graph=build(times,n);
    int[] distTo=dijkstra(k,graph);
    int res=0;
    for(int i=1;i<distTo.length;i++){
        if(distTo[i]==Integer.MAX_VALUE)
            return -1;
        res=Math.max(res,distTo[i]);
    }
    return res;
}
public List<int[]>[] build(int[][] times, int n){
    List<int[]>[] graph=new LinkedList[n+1];
    for(int i=1;i<=n;i++)
        graph[i]=new LinkedList<>();
    for(int[] time:times){
        int u=time[0];
        int v=time[1];
        int w=time[2];
        graph[u].add(new int[]{v,w});
    }
    return graph;
}
class State{
    int id;
    int dist;
    public State(int id,int dist){
        this.id=id;
        this.dist=dist;
    }
}
public int[] dijkstra(int start,List<int[]>[] graph){
    int[] distTo=new int[graph.length];
    Arrays.fill(distTo,Integer.MAX_VALUE);
    distTo[start]=0;

    Queue<State> pq=new PriorityQueue<>((a,b)->{return a.dist-b.dist;});
    pq.offer(new State(start,0));

    while (!pq.isEmpty()){
        State cur=pq.poll();
        int curId=cur.id;
        int curDist=cur.dist;
        if(distTo[curId]<curDist)
            continue;
        for(int[] neighbor:graph[curId]){
            int nextId=neighbor[0];
            int nextDist=neighbor[1]+distTo[curId];
            if(nextDist<distTo[nextId]){
                distTo[nextId]=nextDist;
                pq.offer(new State(nextId,nextDist));
            }
        }
    }
    return distTo;
}
```

## 暴力搜索算法

### 回溯算法速记卡

* 46.全排列

```java
List<List<Integer>> res=new LinkedList<>();
public List<List<Integer>> permute(int[] nums) {
    LinkedList<Integer> track=new LinkedList<>();
    backtrack(nums,track);
    return res;
}
void backtrack(int[] nums,LinkedList<Integer> track){
    if(track.size()==nums.length){
        res.add(new LinkedList(track));
        return;
    }
    for(int i=0;i<nums.length;i++){
        if(track.contains(nums[i]))
            continue;
        track.add(nums[i]);
        backtrack(nums,track);
        track.removeLast();
    }
}
```

* 78.子集

```java
List<List<Integer>> res=new LinkedList<>();
List<Integer> track=new LinkedList<>();
public List<List<Integer>> subsets(int[] nums) {
    backtrack(0,nums);
    return res;
}
public void backtrack(int cur,int[] nums){
    res.add(new LinkedList(track));
    for(int i=cur;i<nums.length;i++){
        track.add(nums[i]);
        backtrack(i+1,nums);
        track.remove(track.size()-1);
    }
}
```

* 77.组合

```java
List<List<Integer>> res=new LinkedList<>();
List<Integer> track=new LinkedList<>();
public List<List<Integer>> combine(int n, int k) {
    if(n<=0||k<=0)
        return res;
    backtrace(1,n,k);
    return res;
}
public void backtrace(int cur,int n,int k){
    if(track.size()==k){
        res.add(new LinkedList<>(track));
        return;
    }
    for(int i=cur;i<=n;i++){
        track.add(i);
        backtrace(i+1,n,k);
        track.remove(track.size()-1);
    }
}
```

* 39.组合总数

```java
List<List<Integer>> ygq=new LinkedList<>();
List<Integer> track=new LinkedList<>();
int sum=0;
public List<List<Integer>> combinationSum(int[] candidates, int target) {
    if(candidates.length==0)
        return ygq;
    backtrace(candidates,target,0);
    return ygq;
}
public void backtrace(int[] candidates, int target,int start){
    if(sum>target)
        return;
    if(sum==target){
        ygq.add(new LinkedList<>(track));
        return;
    }
    for(int i=start;i<candidates.length;i++){
        track.add(candidates[i]);
        sum+=candidates[i];
        backtrace(candidates,target,i);
        sum-=candidates[i];
        track.remove(track.size()-1);
    }
}
```

* 17.电话号码的字母组合

```java
List<String> ygq=new LinkedList<>();
Map<Character,String> map;
public List<String> letterCombinations(String digits) {
    map=new HashMap<>();
    map.put('2',"abc");
    map.put('3',"def");
    map.put('4',"ghi");
    map.put('5',"jkl");
    map.put('6',"mno");
    map.put('7',"pqrs");
    map.put('8',"tuv");
    map.put('9',"wxyz");
    if(digits.length()==0)
        return ygq;
    backtrace(digits,0,new StringBuffer());
    return ygq;
}
public void backtrace(String digits,int start,StringBuffer stringBuffer){
    if(start==digits.length()){
        ygq.add(stringBuffer.toString());
        return;
    }
    char digit=digits.charAt(start);
    String leters=map.get(digit);
    int count=leters.length();
    for(int i=0;i<count;i++){
        stringBuffer.append(leters.charAt(i));
        backtrace(digits,start+1,stringBuffer);
        stringBuffer.deleteCharAt(start);
    }
}
```

* 37.解数独

```java
public void solveSudoku(char[][] board) {
    backtrace(board,0,0);
}
public boolean backtrace(char[][] board,int i,int j){
    if(j==9)
        return backtrace(board,i+1,0);
    if(i==9)
        return true;
    if(board[i][j]!='.')
        return backtrace(board,i,j+1);
    for(char ch='1';ch<='9';ch++){
        if(!isvalid(board,i,j,ch))
            continue;
        board[i][j]=ch;
        if(backtrace(board,i,j+1))
            return true;
        board[i][j]='.';

    }
    return false;
}
public boolean isvalid(char[][] board,int i,int j,char ch){
    for(int k=0;k<9;k++){
        if(board[i][k]==ch)
            return false;
        if(board[k][j]==ch)
            return false;
        if(board[(i/3)*3+k/3][(j/3)*3+k%3]==ch)
            return false;
    }
    return true;
}
```

* 51.N皇后

```java
List<List<String>> ygq=new LinkedList<>();
public List<List<String>> solveNQueens(int n) {
    char[][] board=new char[n][n];
    for(char[] tmp:board)
        Arrays.fill(tmp,'.');
    backtrack(board,0);
    return ygq;
}
public void backtrack(char[][] board,int row){
    int n=board.length;
    if(row==n){
        ygq.add(Tolist(board));
        return;
    }
    for(int col=0;col<n;col++){
        if(!isvalid(board,row,col))
            continue;
        board[row][col]='Q';
        backtrack(board,row+1);
        board[row][col]='.';
    }
}
public boolean isvalid(char[][] board,int row,int col){
    int n=board.length;
    for(int i=0;i<n;i++){
        if(board[i][col]=='Q')
            return false;
    }
    for(int i=row-1,j=col+1;i>=0&&j<n;i--,j++){
        if(board[i][j]=='Q')
            return false;
    }
    for(int i=row-1,j=col-1;i>=0&&j>=0;i--,j--){
        if(board[i][j]=='Q')
            return false;
    }
    return true;
}
public List<String> Tolist(char[][] board){
    List<String> list=new LinkedList<>();
    for(char[] tmp:board){
        list.add(String.copyValueOf(tmp));
    }
    return list;
}
```

* 22.括号生成

```java
List<String> ygq=new LinkedList<>();
public List<String> generateParenthesis(int n) {
    backtrack(0,0,n,new StringBuffer());
    return ygq;
}
public void backtrack(int left,int right,int n,StringBuffer stringBuffer){
    if(stringBuffer.length()==n*2){
        ygq.add(stringBuffer.toString());
        return;
    }
    if(left<n){
        stringBuffer.append('(');
        backtrack(left+1,right,n,stringBuffer);
        stringBuffer.deleteCharAt(stringBuffer.length()-1);
    }
    if(right<left){
        stringBuffer.append(')');
        backtrack(left,right+1,n,stringBuffer);
        stringBuffer.deleteCharAt(stringBuffer.length()-1);
    }
}
```

* 494.目标和

```java
int sum=0;
int count=0;
public int findTargetSumWays(int[] nums, int target) {
    backtrack(nums, target,0);
    return count;
}
public void backtrack(int[] nums,int target,int start){
    if(sum==target&&start==nums.length){
        count++;
        return;
    }
    if(start==nums.length){
        return;
    }
    sum+=nums[start];
    backtrack(nums,target,start+1);
    sum-=nums[start];

    sum-=nums[start];
    backtrack(nums,target,start+1);
    sum+=nums[start];
}
```

### DFS算法速记卡

* 200.岛屿数量

```java
public int numIslands(char[][] grid) {
    int m=grid.length;
    int n=grid[0].length;
    UF uf=new UF(m*n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]=='1'){
                if(i-1>=0&&grid[i-1][j]=='1')
                    uf.union((i-1)*n+j,i*n+j);
                if(i+1<m&&grid[i+1][j]=='1')
                    uf.union((i+1)*n+j,i*n+j);
                if(j-1>=0&&grid[i][j-1]=='1')
                    uf.union(i*n+j-1,i*n+j);
                if(j+1<n&&grid[i][j+1]=='1')
                    uf.union(i*n+j+1,i*n+j);
            }
        }
    }
    Set<Integer> set=new HashSet<>();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]=='1'){
                set.add(uf.find(i*n+j));
            }
        }
    }
    return set.size();
}
class UF{
    private int count;
    private int[] partent;
    private int[] size;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<n;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]<size[rootq]){
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }else{
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (x!=partent[x]){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
    public int getCount(){
        return this.count;
    }
}
```

* 1254.统计封闭岛屿的数目

```java
public int closedIsland(int[][] grid) {
    int m=grid.length;
    int n=grid[0].length;
    int dummy=m*n;
    UF uf=new UF(m*n+1);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==0){
                if(i-1>=0&&grid[i-1][j]==0){
                    uf.union(i*n+j,(i-1)*n+j);
                }
                if(i+1<m&&grid[i+1][j]==0){
                    uf.union(i*n+j,(i+1)*n+j);
                }
                if(j-1>=0&&grid[i][j-1]==0){
                    uf.union(i*n+j,i*n+j-1);
                }
                if(j+1<n&&grid[i][j+1]==0){
                    uf.union(i*n+j,i*n+j+1);
                }
            }
        }
    }
    for(int i=0;i<m;i++){
        if(grid[i][0]==0)
            uf.union(i*n+0,dummy);
        if(grid[i][n-1]==0)
            uf.union(i*n+n-1,dummy);
    }
    for(int j=0;j<n;j++){
        if(grid[0][j]==0)
            uf.union(0*n+j,dummy);
        if(grid[m-1][j]==0)
            uf.union((m-1)*n+j,dummy);
    }
    Set<Integer> set=new HashSet<>();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==0){
                set.add(uf.find(i*n+j));
            }
        }
    }
    set.add(uf.find(dummy));
    return set.size()-1;
}
class UF{
    private int[] partent;
    private int[] size;
    private int count;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<count;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]<size[rootq]){
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }else {
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (x!=partent[x]){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
}
```

* 1020.飞地的数量

```java
public int numEnclaves(int[][] grid) {
    int count=0;
    int m=grid.length;
    int n=grid[0].length;
    UF uf=new UF(m*n+1);
    int dummy=m*n;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==1){
                if(i-1>=0&&grid[i-1][j]==1)
                    uf.union(i*n+j,(i-1)*n+j);
                if(i+1<m&&grid[i+1][j]==1)
                    uf.union(i*n+j,(i+1)*n+j);
                if(j-1>=0&&grid[i][j-1]==1)
                    uf.union(i*n+j,i*n+j-1);
                if(j+1<n&&grid[i][j+1]==1)
                    uf.union(i*n+j,i*n+j+1);
            }
        }
    }
    for(int i=0;i<m;i++){
        if(grid[i][0]==1){
            uf.union(i*n+0,dummy);
        }
        if(grid[i][n-1]==1){
            uf.union(i*n+n-1,dummy);
        }
    }
    for(int j=0;j<n;j++){
        if(grid[0][j]==1)
            uf.union(0*n+j,dummy);
        if(grid[m-1][j]==1)
            uf.union((m-1)*n+j,dummy);
    }
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==1&&!uf.connected(i*n+j,dummy))
                count++;
        }
    }
    return count;
}
class UF{
    private int[] partent;
    private int[] size;
    private int count;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<count;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]<size[rootq]){
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }else {
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (x!=partent[x]){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
}
```

* 695.岛屿的最大面积

```java
public int maxAreaOfIsland(int[][] grid) {
    int max=0;
    int m=grid.length;
    int n=grid[0].length;
    UF uf=new UF(m*n);
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==1){
                if(i-1>=0&&grid[i-1][j]==1){
                    uf.union(i*n+j,(i-1)*n+j);
                }
                if(i+1<m&&grid[i+1][j]==1){
                    uf.union(i*n+j,(i+1)*n+j);
                }
                if(j-1>=0&&grid[i][j-1]==1){
                    uf.union(i*n+j,i*n+j-1);
                }
                if(j+1<n&&grid[i][j+1]==1){
                    uf.union(i*n+j,i*n+j+1);
                }
            }
        }
    }
    Map<Integer,Integer> map=new HashMap<>();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid[i][j]==1){
                int tmp=uf.find(i*n+j);
                map.put(tmp,map.getOrDefault(tmp,0)+1);
            }
        }
    }
    for(int tmp:map.values())
        max=Math.max(max,tmp);

    return max;
}
class UF{
    private int[] partent;
    private int[] size;
    private int count;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<count;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]<size[rootq]){
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }else {
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (x!=partent[x]){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
}
```

* 1905.统计子岛屿

```java
public int countSubIslands(int[][] grid1, int[][] grid2){
    int max=0;
    int m=grid1.length;
    int n=grid1[0].length;
    UF uf=new UF(m*n+1);
    int dummy=m*n;
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid1[i][j]==0&&grid2[i][j]==1)
                uf.union(i*n+j,dummy);
            if(grid2[i][j]==1){
                if(i-1>=0&&grid2[i-1][j]==1){
                    uf.union(i*n+j,(i-1)*n+j);
                }
                if(i+1<m&&grid2[i+1][j]==1){
                    uf.union(i*n+j,(i+1)*n+j);
                }
                if(j-1>=0&&grid2[i][j-1]==1){
                    uf.union(i*n+j,i*n+j-1);
                }
                if(j+1<n&&grid2[i][j+1]==1){
                    uf.union(i*n+j,i*n+j+1);
                }
            }
        }
    }
    Set<Integer> set=new HashSet<>();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(grid2[i][j]==1&&!uf.connected(i*n+j,dummy)){
                int tmp=uf.find(i*n+j);
                set.add(tmp);
            }
        }
    }
    return set.size();
}
class UF{
    private int[] partent;
    private int[] size;
    private int count;
    public UF(int n){
        this.count=n;
        partent=new int[n];
        size=new int[n];
        for(int i=0;i<count;i++){
            partent[i]=i;
            size[i]=1;
        }
    }
    public void union(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        if(rootp==rootq)
            return;
        if(size[rootp]<size[rootq]){
            partent[rootp]=rootq;
            size[rootq]+=size[rootp];
        }else {
            partent[rootq]=rootp;
            size[rootp]+=size[rootq];
        }
        count--;
    }
    public boolean connected(int p,int q){
        int rootp=find(p);
        int rootq=find(q);
        return rootp==rootq;
    }
    public int find(int x){
        while (x!=partent[x]){
            partent[x]=partent[partent[x]];
            x=partent[x];
        }
        return x;
    }
}
```

### BFS算法速记卡

* 752.打开转盘锁

```java
public int openLock(String[] deadends, String target) {
    Set<String> deads=new HashSet<>();
    for(String s:deadends)
        deads.add(s);
    Set<String> visited=new HashSet<>();
    Queue<String> queue=new LinkedList<>();
    queue.add("0000");
    visited.add("0000");
    int step=0;
    while (!queue.isEmpty()){
        int size=queue.size();
        for(int i=0;i<size;i++){
            String cur=queue.poll();
            if(deads.contains(cur))
                continue;
            if(cur.equals(target))
                return step;
            for(int j=0;j<4;j++){
                String up=plusOne(cur,j);
                if(!visited.contains(up)){
                    queue.offer(up);
                    visited.add(up);
                }
                String down=minusOne(cur,j);
                if(!visited.contains(down)){
                    queue.offer(down);
                    visited.add(down);
                }
            }
        }
        step++;
    }
    return -1;
}
public String plusOne(String cur,int j){
    char[] ch=cur.toCharArray();
    if(ch[j]=='9')
        ch[j]='0';
    else
        ch[j]+=1;
    return new String(ch);
}
public String minusOne(String cur,int j){
    char[] ch=cur.toCharArray();
    if(ch[j]=='0')
        ch[j]='9';
    else
        ch[j]-=1;
    return new String(ch);
}
```

* 773.滑动谜题

```java
public int slidingPuzzle(int[][] board) {
    int m=2,n=3;
    StringBuffer stringBuffer=new StringBuffer();
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++)
            stringBuffer.append(board[i][j]);
    }
    String start=stringBuffer.toString();
    String target="123450";
    Queue<String> queue=new LinkedList<>();
    Set<String> visited=new HashSet<>();
    queue.add(start);
    visited.add(start);
    int step=0;
    int[][] neighbor=new int[][]{{1,3},{0,2,4},{1,5},{0,4},{1,3,5},{2,4}};
    while (!queue.isEmpty()){
        int size=queue.size();
        for(int i=0;i<size;i++){
            String cur=queue.poll();
            if(cur.equals(target))
                return step;
            int idx=0;
            for(;cur.charAt(idx)!='0';idx++);
            for(int idj:neighbor[idx]){
                String tmp=swap(cur,idx,idj);
                if(!visited.contains(tmp)){
                    queue.offer(tmp);
                    visited.add(tmp);
                }
            }
        }
        step++;
    }
    return -1;
}
public String swap(String cur,int idx,int idj){
    char[] ch=cur.toCharArray();
    char tmp=ch[idx];
    ch[idx]=ch[idj];
    ch[idj]=tmp;
    return new String(ch);
}
```

## 动态规划算法

### 一维动态规划速记卡

* 70.爬楼梯

```java
int[] dp;
public int climbStairs(int n) {
    dp=new int[n+1];
    if(n<=2)
        return n;
    dp[1]=1;
    dp[2]=2;
    for(int i=3;i<=n;i++){
        dp[i]=dp[i-1]+dp[i-2];
    }
    return dp[n];
}
```

* 322.零钱兑换

```java
public int coinChange(int[] coins, int amount) {
    int n=coins.length;
    int[] dp=new int[amount+1];
    Arrays.fill(dp,amount+1);
    dp[0]=0;
    for(int i=1;i<=amount;i++){
        for(int j=0;j<n;j++){
            if(coins[j]<=i)
                dp[i]=Math.min(dp[i],dp[i-coins[j]]+1);
        }
    }
    return dp[amount]>amount?-1:dp[amount];
}
```

* 300.最长递增子序列

```java
public int lengthOfLIS(int[] nums) {
    int count=0;
    int[] dp=new int[nums.length];
    Arrays.fill(dp,1);
    for(int i=0;i<dp.length;i++){
        for(int j=0;j<i;j++){
            if(nums[j]<nums[i])
                dp[i]=Math.max(dp[i],dp[j]+1);
        }
        count=Math.max(count,dp[i]);
    }
    return count;
}
```

* 53.最大子数组和

```java
public int maxSubArray(int[] nums) {
    int n=nums.length;
    int[] dp=new int[n];
    dp[0]=nums[0];
    int max=nums[0];
    for(int i=1;i<n;i++){
        dp[i]=Math.max(nums[i],dp[i-1]+nums[i]);
        max=Math.max(dp[i],max);
    }
    return max;
}
```

* 55.跳跃游戏

```java
public boolean canJump(int[] nums) {
    int n=nums.length;
    int max=0;
    for(int i=0;i<n-1;i++){
        max=Math.max(max,nums[i]+i);
        if(max<=i)
            return false;
    }
    return max>=n-1;
}
```

* 45.跳跃游戏II

```java
public int jump(int[] nums) {
    int count=0;
    int n=nums.length;
    int max=0;
    int cur=0;
    for(int i=0;i<n-1;i++){
        max=Math.max(max,nums[i]+i);
        if(cur==i){
            count++;
            cur=max;
        }
    }
    return count;
}
```

* 198.打家劫舍

```java
public int rob(int[] nums) {
    int n=nums.length;
    int[] dp=new int[n];
    if(n==1)
        return nums[0];
    if(n==2)
        return Math.max(nums[0],nums[1]);
    dp[0]=nums[0];
    dp[1]=Math.max(nums[0],nums[1]);
    for(int i=2;i<n;i++){
        dp[i]=Math.max(nums[i]+dp[i-2],dp[i-1]);
    }
    return dp[n-1];
}
```

* 213.打家劫舍II

```java
public int rob(int[] nums) {
    int n=nums.length;
    if(n==1)
        return nums[0];
    if(n==2)
        return Math.max(nums[0],nums[1]);
    if(n==3){
        return Math.max(Math.max(nums[0],nums[1]),nums[2]);
    }
    int[] dp=new int[n];
    dp[0]=nums[0];
    dp[1]=Math.max(nums[0],nums[1]);
    int[] dp1=new int[n];
    dp1[0]=0;dp1[1]=nums[1];
    dp1[2]=Math.max(nums[1],nums[2]);
    for(int i=2;i<n;i++){
        dp[i]=Math.max(dp[i-1],nums[i]+dp[i-2]);
    }
    for(int i=3;i<n;i++){
        dp1[i]=Math.max(dp1[i-1],nums[i]+dp1[i-2]);
    }
    return Math.max(dp[n-2],dp1[n-1]);
}
```

* 337.打家劫舍III

```java
public int rob(TreeNode root) {
    int[] ans=ygq(root);
    return Math.max(ans[0],ans[1]);
}
public int[] ygq(TreeNode root){
     if(root==null)
         return new int[]{0,0};//根结点在最大值内，根结点不在最大值内
     int[] left=ygq(root.left);
     int[] right=ygq(root.right);
     int max1=left[0]+right[0];
     int max2=left[1]+right[1]+root.val;
     max2=Math.max(max1,max2);
     return new int[]{max2,max1};//注意顺序
}
```

### 二维动态规划速记卡

* 64.最小路径和

```java
public int minPathSum(int[][] grid) {
    int m=grid.length;
    int n=grid[0].length;
    int[][] dp=new int[m][n];
    dp[0][0]=grid[0][0];
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(j-1>=0&&i-1>=0)
                dp[i][j]=grid[i][j]+Math.min(dp[i][j-1],dp[i-1][j]);
            else if(j-1>=0)
                dp[i][j]=grid[i][j]+dp[i][j-1];
            else if(i-1>=0)
                dp[i][j]=grid[i][j]+dp[i-1][j];
            else
                continue;
        }
    }
    return dp[m-1][n-1];
}
```

* 62.不同路径

```java
public int uniquePaths(int m, int n) {
    int[][] dp=new int[m][n];
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            if(i==0||j==0)
                dp[i][j]=1;
            else
                dp[i][j]=dp[i-1][j]+dp[i][j-1];
        }
    }
    return dp[m-1][n-1];
}
```

* 931.下降路径最小和

```java
public int minFallingPathSum(int[][] matrix) {
    int n=matrix.length;
    int[][] dp=new int[n][n];
    int count=Integer.MAX_VALUE;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==0)
                dp[i][j]=matrix[i][j];
            else if(j-1<0)
                dp[i][j]=matrix[i][j]+Math.min(dp[i-1][j],dp[i-1][j+1]);
            else if(j+1==n)
                dp[i][j]=matrix[i][j]+Math.min(dp[i-1][j],dp[i-1][j-1]);
            else
                dp[i][j]=matrix[i][j]+Math.min(dp[i-1][j],Math.min(dp[i-1][j-1],dp[i-1][j+1]));
        }
    }
    for(int i=0;i<n;i++)
        count=Math.min(count,dp[n-1][i]);
    return count;
}
```

* 72.编辑距离

```java
public int minDistance(String word1, String word2) {
    int m=word1.length();
    int n=word2.length();
    if(m==0)
        return n;
    if(n==0)
        return m;
    int[][] dp=new int[m+1][n+1];
    for(int i=1;i<=m;i++)
        dp[i][0]=i;
    for(int j=1;j<=n;j++)
        dp[0][j]=j;
    for(int i=1;i<=m;i++){
        for(int j=1;j<=n;j++){
            if(word1.charAt(i-1)==word2.charAt(j-1))
                dp[i][j]=dp[i-1][j-1];
            else
                dp[i][j]=Math.min(Math.min(dp[i][j-1],dp[i-1][j]),dp[i-1][j-1])+1;
        }
    }
    return dp[m][n];
}
```

* 10.正则表达式匹配

```java
public boolean isMatch(String s, String p) {
    char[] cs = s.toCharArray();
    char[] cp = p.toCharArray();

    // dp[i][j]:表示s的前i个字符，p的前j个字符是否能够匹配
    boolean[][] dp = new boolean[cs.length + 1][cp.length + 1];

    // 初期值
    // s为空，p为空，能匹配上
    dp[0][0] = true;
    // p为空，s不为空，必为false(boolean数组默认值为false，无需处理)

    // s为空，p不为空，由于*可以匹配0个字符，所以有可能为true
    for (int j = 1; j <= cp.length; j++) {
        if (cp[j - 1] == '*') {
            dp[0][j] = dp[0][j - 2];
        }
    }

    // 填格子
    for (int i = 1; i <= cs.length; i++) {
        for (int j = 1; j <= cp.length; j++) {
            // 文本串和模式串末位字符能匹配上
            if (cs[i - 1] == cp[j - 1] || cp[j - 1] == '.') {
                dp[i][j] = dp[i - 1][j - 1];
            } else if (cp[j - 1] == '*') { // 模式串末位是*
                // 模式串*的前一个字符能够跟文本串的末位匹配上
                if (cs[i - 1] == cp[j - 2] || cp[j - 2] == '.') {
                    dp[i][j] = dp[i][j - 2]      // *匹配0次的情况
                            || dp[i - 1][j];     // *匹配1次或多次的情况
                } else { // 模式串*的前一个字符不能够跟文本串的末位匹配
                    dp[i][j] = dp[i][j - 2];     // *只能匹配0次
                }
            }
        }
    }
    return dp[cs.length][cp.length];
}
```

* 121.买卖股票的最佳时机

```java
public int maxProfit(int[] prices) {
    int n=prices.length;
    int[][] dp=new int[n][2];
    for(int i=0;i<n;i++){
        if(i==0){
            dp[i][0]=0;
            dp[i][1]=-prices[i];
            continue;
        }
        dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
        dp[i][1]=Math.max(dp[i-1][1],-prices[i]);//notice, notice
    }
    return dp[n-1][0];
}
```

* 122.买卖股票的最佳时机II

```java
public int maxProfit(int[] prices) {
    int n=prices.length;
    int[][] dp=new int[n][2];
    for(int i=0;i<n;i++){
        if(i==0){
            dp[i][0]=0;
            dp[i][1]=-prices[i];
            continue;
        }
        dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
        dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]);
    }
    return dp[n-1][0];
}
```

* 123.买卖股票的最佳时机III

```java
public int maxProfit(int[] prices) {
    int kmax=2,n=prices.length;
    int[][][] dp=new int[n][kmax+1][2];
    for(int i=0;i<n;i++){
        for(int k=kmax;k>=1;k--){
            if(i==0){
                dp[i][k][0]=0;
                dp[i][k][1]=-prices[i];
                continue;
            }
            dp[i][k][0]=Math.max(dp[i-1][k][0],dp[i-1][k][1]+prices[i]);
            dp[i][k][1]=Math.max(dp[i-1][k][1],dp[i-1][k-1][0]-prices[i]);
        }
    }
    return dp[n-1][kmax][0];
}
```

* 188.买卖股票的最佳时机IV

```java
public int maxProfit(int k, int[] prices) {
    int n=prices.length;
    if(n==0)
        return 0;
    if(k>n/2)
        return kmaxProfit(prices);
    int[][][] dp=new int[n][k+1][2];
    for(int i=0;i<n;i++){
        for(int j=k;j>=1;j--){
            if(i==0){
                dp[i][j][0]=0;
                dp[i][j][1]=-prices[i];
                continue;
            }
            dp[i][j][0]=Math.max(dp[i-1][j][0],dp[i-1][j][1]+prices[i]);
            dp[i][j][1]=Math.max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i]);
        }
    }
    return dp[n-1][k][0];
}
public int kmaxProfit(int[] prices){
    int n=prices.length;
    int[][] dp=new int[n][2];
    for(int i=0;i<n;i++){
        if(i==0){
            dp[i][0]=0;
            dp[i][1]=-prices[i];
            continue;
        }
        dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
        dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]);
    }
    return dp[n-1][0];
}
```

* 309.最佳买卖股票时机含冷冻期

```java
public int maxProfit(int[] prices) {
    int n=prices.length;
    int[][] dp=new int[n][2];
    for(int i=0;i<n;i++){
        if(i==0){
            dp[i][0]=0;
            dp[i][1]=-prices[i];
            continue;
        }
        if(i==1){
            dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
            dp[i][1]=Math.max(dp[i-1][1],-prices[i]);
            continue;
        }
        dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
        dp[i][1]=Math.max(dp[i-1][1],dp[i-2][0]-prices[i]);
    }
    return dp[n-1][0];
}
```

* 714.买卖股票的最佳时机含手续费

```java
public int maxProfit(int[] prices, int fee) {
    int n=prices.length;
    int[][] dp=new int[n][2];
    for(int i=0;i<n;i++){
        if(i==0){
            dp[i][0]=0;
            dp[i][1]=-prices[i]-fee;
            continue;
        }
        dp[i][0]=Math.max(dp[i-1][0],dp[i-1][1]+prices[i]);
        dp[i][1]=Math.max(dp[i-1][1],dp[i-1][0]-prices[i]-fee);
    }
    return dp[n-1][0];
}
```

* 174.地下城游戏

```java
public int calculateMinimumHP(int[][] dungeon) {
    int m=dungeon.length;
    int n=dungeon[0].length;
    int[][] dp=new int[m][n];
    dp[m-1][n-1]=Math.max(0,-dungeon[m-1][n-1]);
    for(int i=m-2;i>=0;i--){
        int tmp=dp[i+1][n-1]-dungeon[i][n-1];
        dp[i][n-1]=Math.max(0,tmp);
    }
    for(int j=n-2;j>=0;j--){
        int tmp=dp[m-1][j+1]-dungeon[m-1][j];
        dp[m-1][j]=Math.max(0,tmp);
    }
    for(int i=m-2;i>=0;i--){
        for(int j=n-2;j>=0;j--){
            int tmp=Math.min(dp[i+1][j],dp[i][j+1]);
            tmp=tmp-dungeon[i][j];
            dp[i][j]=Math.max(0,tmp);
        }
    }
    return dp[0][0]+1;
}
```

* 剑指 Offer II 119. 最长连续序列

```java
public int longestConsecutive(int[] nums) {
    int n=nums.length;
    Arrays.sort(nums);
    if(n==0)
        return 0;
    int count=1;
    int[] dp=new int[n];
    Arrays.fill(dp,1);
    for(int i=1;i<n;i++){
        if(nums[i]==nums[i-1]+1){
            dp[i]=dp[i-1]+1;
            count=Math.max(count,dp[i]);
        }else if(nums[i]==nums[i-1]){
            dp[i]=dp[i-1];
        }
    }
    return count;
}
```

* 面试题 10.03. 搜索旋转数组

```java
public int search(int[] arr, int target) {
    Map<Integer,Integer> map=new HashMap<>();
    for(int i=0;i<arr.length;i++){
        if(!map.containsKey(arr[i]))
            map.put(arr[i],i);
    }
    Arrays.sort(arr);
    int left=0,right=arr.length-1;
    while (left<=right){
        int mid=left+(right-left)/2;
        if(arr[mid]<target)
            left=mid+1;
        else if(arr[mid]>target)
            right=mid-1;
        else
            return map.get(arr[mid]);
    }
    return -1;
}
```

* 312.戳气球

```java
public int maxCoins(int[] nums) {
    int n=nums.length;
    int[] tmp=new int[n+2];
    tmp[0]=1;
    tmp[n+1]=1;
    for(int i=0;i<n;i++)
        tmp[i+1]=nums[i];
    int[][] dp=new int[n+2][n+2];
    for(int len=3;len<=n+2;len++){
        for(int i=0;i<=n+2-len;i++){
            int res=0;
            for(int k=i+1;k<i+len-1;k++){
                int left=dp[i][k];
                int right=dp[k][i+len-1];
                res=Math.max(res,left+tmp[i]*tmp[k]*tmp[i+len-1]+right);
            }
            dp[i][i+len-1]=res;
        }
    }
    return dp[0][n+1];
}
```

### 背包问题速记卡

* 零钱兑换II

```java
public int change(int amount, int[] coins) {
    int n=coins.length;
    int[][] dp=new int[n+1][amount+1];
    dp[0][0]=1;
    for(int i=1;i<=n;i++){
        int val=coins[i-1];
        for(int j=0;j<=amount;j++){
            dp[i][j]=dp[i-1][j];
            for(int k=1;k*val<=j;k++){
                dp[i][j]+=dp[i-1][j-k*val];
            }
        }
    }
    return dp[n][amount];
}
```
