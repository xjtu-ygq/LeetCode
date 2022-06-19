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

## 滑动窗口速记卡

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

## 其他双指针算法速记卡

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

## 链表双指针速记卡

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

## 前缀和技巧速记卡

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
