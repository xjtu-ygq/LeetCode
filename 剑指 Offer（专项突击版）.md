# 剑指 Offer（专项突击版）

* 剑指 Offer II 001. 整数除法

```java
public int divide(int a, int b) {
    int res=0;
    if(a==Integer.MIN_VALUE&&b==-1)
        return Integer.MAX_VALUE;
    int sign=(a>0)^(b>0)?-1:1;
    a=Math.abs(a);
    b=Math.abs(b);
    for(int i=31;i>=0;i--){
        if((a>>>i)-b>=0){
            a-=(b<<i);
            res+=(1<<i);
        }
    }
    return res*sign;
}
```

* 剑指 Offer II 002. 二进制加法

```java
public String addBinary(String a, String b) {
    StringBuffer stringBuffer=new StringBuffer();
    int len=Math.max(a.length(),b.length());
    int carry=0;
    for(int i=0;i<len;i++){
        carry+=i<a.length()?(a.charAt(a.length()-i-1)-'0'):0;
        carry+=i<b.length()?(b.charAt(b.length()-i-1)-'0'):0;
        stringBuffer.append((char)(carry%2+'0'));
        carry=carry/2;
    }
    if(carry>0)
        stringBuffer.append('1');
    return stringBuffer.reverse().toString();
}
```

* 剑指 Offer II 003. 前 n 个数字二进制中 1 的个数

```java
public int[] countBits(int n) {
    int[] res=new int[n+1];
    for(int i=0;i<=n;i++){
        int tmp=i;
        res[i]+=i%2;
        for(int j=15;j>=0;j--){
            if((tmp>>j)-2>=0){
                res[i]+=1;
                tmp-=(2<<j);
            }
        }
    }
    return res;
}
```

* 剑指 Offer II 004. 只出现一次的数字

```java
public int singleNumber(int[] nums) {
    Map<Integer,Integer> map=new HashMap<>();
    for(int num:nums){
        map.put(num,map.getOrDefault(num,0)+1);
    }
    for(int key:map.keySet()){
        if(map.get(key)==1){
            return key;
        }
    }
    return 0;
}
```

* 剑指 Offer II 005. 单词长度的最大乘积

```java
public int maxProduct(String[] words) {
    int res=0;
    int n=words.length;
    for(int i=0;i<n;i++){
        String word1=words[i];
        for(int j=i+1;j<n;j++){
            String word2=words[j];
            if(!hasSameChar(word1,word2)){
                res=Math.max(res,word1.length()*word2.length());
            }
        }
    }
    return res;
}
public boolean hasSameChar(String word1,String word2){
    for(char c:word1.toCharArray()){
        if(word2.indexOf(c)!=-1)
            return true;
    }
    return false;
}
```

* 剑指 Offer II 006. 排序数组中两个数字之和

```java
public int[] twoSum(int[] numbers, int target) {
    int res[]=new int[2];
    int left=0,right=numbers.length-1;
    while (left<right){
        int sum=numbers[left]+numbers[right];
        if(sum<target){
            left++;
        }else if(sum>target){
            right--;
        }else {
            res[0]=left;
            res[1]=right;
            return res;
        }
    }
    return res;
}
```

* 剑指 Offer II 007. 数组中和为 0 的三个数

```java
public List<List<Integer>> threeSum(int[] nums) {
    List<List<Integer>> res=new ArrayList<>();
    Arrays.sort(nums);
    for(int i=0;i<nums.length;i++){
        List<List<Integer>> tmps=twoSum(nums,i+1,0-nums[i]);
        for(List<Integer> tmp:tmps){
            tmp.add(nums[i]);
            res.add(tmp);
        }
        while (i<nums.length-1&&nums[i]==nums[i+1])i++;
    }
    return res;
}
public List<List<Integer>> twoSum(int[] nums,int start,int target){
    List<List<Integer>> res=new ArrayList<>();
    int left=start,right=nums.length-1;
    while (left<right){
        int n1=nums[left],n2=nums[right];
        int sum=n1+n2;
        if(sum<target){
            while (left<right&&nums[left]==n1)left++;
        }else if(sum>target){
            while (left<right&&nums[right]==n2)right--;
        }else {
            List<Integer> tmp=new ArrayList<>();
            tmp.add(nums[left]);tmp.add(nums[right]);res.add(tmp);
            while (left<right&&nums[left]==n1)left++;
            while (left<right&&nums[right]==n2)right--;
        }
    }
    return res;
}
```

* 剑指 Offer II 009. 乘积小于 K 的子数组

```java
public int numSubarrayProductLessThanK(int[] nums, int k) {
    int ans=0;
    int left=0,right=0;
    int sum=1;
    while (right<nums.length){
        sum*=nums[right++];
        while (sum>=k&&left<right){
            sum=sum/nums[left];
            left++;
        }
        ans+=right-left;
    }
    return ans;
}
```

* 剑指 Offer II 010. 和为 k 的子数组

```java
public int subarraySum(int[] nums, int k) {
    int ans=0;
    int n=nums.length;
    int dp[]=new int[n+1];
    dp[0]=0;
    for(int i=1;i<=n;i++){
        dp[i]=dp[i-1]+nums[i-1];
    }
    for(int i=0;i<n;i++){
        for(int j=i+1;j<=n;j++){
            if(dp[j]-dp[i]==k)
                ans++;
        }
    }
    return ans;
}
```

* 剑指 Offer II 011. 0 和 1 个数相同的子数组

```java
public int findMaxLength(int[] nums) {
    int maxLength = 0;
    Map<Integer, Integer> map = new HashMap<Integer, Integer>();
    int counter = 0;
    map.put(counter, -1);
    int n = nums.length;
    for (int i = 0; i < n; i++) {
        int num = nums[i];
        if (num == 1) {
            counter++;
        } else {
            counter--;
        }
        if (map.containsKey(counter)) {
            int prevIndex = map.get(counter);
            maxLength = Math.max(maxLength, i - prevIndex);
        } else {
            map.put(counter, i);
        }
    }
    return maxLength;
}
```

* 剑指 Offer II 012. 左右两边子数组的和相等

```java
public int pivotIndex(int[] nums) {
    int total=Arrays.stream(nums).sum();
    int sum=0;
    for(int i=0;i<nums.length;i++){
        if(2*sum+nums[i]==total)
            return i;
        sum+=nums[i];
    }
    return -1;
}
```

* 剑指 Offer II 013. 二维子矩阵的和

```java
private int[][] ans;
public NumMatrix(int[][] matrix) {
    ans=new int[matrix.length+1][matrix[0].length+1];
    for(int i=0;i<ans.length;i++){
        ans[i][0]=0;
    }
    for(int j=0;j<ans[0].length;j++){
        ans[0][j]=0;
    }
    for(int i=1;i<ans.length;i++){
        for(int j=1;j<ans[0].length;j++){
            ans[i][j]=matrix[i-1][j-1]+ans[i-1][j]+ans[i][j-1]-ans[i-1][j-1];
        }
    }
}

public int sumRegion(int row1, int col1, int row2, int col2) {
    return ans[row2+1][col2+1]-ans[row1][col2+1]-ans[row2+1][col1]+ans[row1][col1];
}
```

* 剑指 Offer II 014. 字符串中的变位词

```java
public boolean checkInclusion(String s1, String s2) {
    Map<Character,Integer> window=new HashMap<>();
    Map<Character,Integer> need=new HashMap<>();
    for(char c:s1.toCharArray()){
        need.put(c,need.getOrDefault(c,0)+1);
    }
    int left=0,right=0;
    int valid=0;
    while (right<s2.length()){
        char c=s2.charAt(right);
        right++;
        if(need.containsKey(c)){
            window.put(c,window.getOrDefault(c,0)+1);
            if(window.get(c).equals(need.get(c)))
                valid++;
        }
        while (right-left>=s1.length()){
            if(valid==need.size())
                return true;
            char d=s2.charAt(left);
            left++;
            if(need.containsKey(d)){
                if(need.get(d).equals(window.get(d))){
                    valid--;
                }
                window.put(d,window.get(d)-1);
            }
        }
    }
    return false;
}
```

* 剑指 Offer II 015. 字符串中的所有变位词

```java
public List<Integer> findAnagrams(String s, String p) {
    List<Integer> ans=new ArrayList<>();
    Map<Character,Integer> window=new HashMap<>();
    Map<Character,Integer> need=new HashMap<>();
    for(char c:p.toCharArray()){
        need.put(c,need.getOrDefault(c,0)+1);
    }
    int left=0,right=0;
    int valid=0;
    while (right<s.length()){
        char c=s.charAt(right);
        right++;
        if(need.containsKey(c)){
            window.put(c,window.getOrDefault(c,0)+1);
            if(window.get(c).equals(need.get(c))){
                valid++;
            }
        }
        while (right-left>=p.length()){
            if(valid==need.size()){
                ans.add(left);
            }
            char d=s.charAt(left);
            left++;
            if(need.containsKey(d)){
                if(need.get(d).equals(window.get(d))){
                    valid--;
                }
                window.put(d,window.get(d)-1);
            }
        }
    }
    return ans;
}
```

* 剑指 Offer II 016. 不含重复字符的最长子字符串

```java
public int lengthOfLongestSubstring(String s) {
    int ans=0;
    Map<Character,Integer> map=new HashMap<>();
    int left=0,right=0;
    while (right<s.length()){
        char c=s.charAt(right);
        right++;
        map.put(c,map.getOrDefault(c,0)+1);
        while (map.get(c)>1){
            char d=s.charAt(left);
            left++;
            map.put(d,map.get(d)-1);
        }
        ans=Math.max(ans,right-left);
    }
    return ans;
}
```

* 剑指 Offer II 017. 含有所有字符的最短字符串

```java
public String minWindow(String s, String t) {
    Map<Character,Integer> window=new HashMap<>();
    Map<Character,Integer> need=new HashMap<>();
    for(char c:t.toCharArray()){
        need.put(c,need.getOrDefault(c,0)+1);
    }
    int left=0,right=0;
    int valid=0;
    int len=Integer.MAX_VALUE,start=0;
    while (right<s.length()){
        char c=s.charAt(right);
        right++;
        if(need.containsKey(c)){
            window.put(c,window.getOrDefault(c,0)+1);
            if(window.get(c).equals(need.get(c))){
                valid++;
            }
        }
        while (valid==need.size()){
            if(right-left<len){
                len=right-left;
                start=left;
            }
            char d=s.charAt(left);
            left++;
            if(need.containsKey(d)){
                if(need.get(d).equals(window.get(d))){
                    valid--;
                }
                window.put(d,window.get(d)-1);
            }
        }
    }
    return len==Integer.MAX_VALUE?"":s.substring(start,start+len);
}
```

* 剑指 Offer II 018. 有效的回文

```java
public boolean isPalindrome(String s) {
    int left=0,right=s.length()-1;
    while (left<right){
        while (left<right&&!Character.isLetterOrDigit(s.charAt(left))){
            left++;
        }
        while (left<right&&!Character.isLetterOrDigit(s.charAt(right))){
            right--;
        }
        if(left<right){
            if(Character.toLowerCase(s.charAt(left))!=Character.toLowerCase(s.charAt(right))){
                System.out.println(s.charAt(left));
                System.out.println(s.charAt(right));
                return false;
            }
            left++;
            right--;
        }
    }
    return true;
}
```

* 剑指 Offer II 019. 最多删除一个字符得到回文

```java
public boolean validPalindrome(String s) {
    int left=0,right=s.length()-1;
    while (left<right){
        if(s.charAt(left)!=s.charAt(right)){
            if(isPalindrome(s.substring(left,right))||isPalindrome(s.substring(left+1,right+1))){
                return true;
            }else
                return false;
        }
        left++;
        right--;
    }
    return true;
}
public boolean isPalindrome(String s){
    int left=0,right=s.length()-1;
    while (left<right){
        if(s.charAt(left)!=s.charAt(right)){
            return false;
        }
        left++;
        right--;
    }
    return true;
}
```

* 剑指 Offer II 020. 回文子字符串的个数

```java
public int countSubstrings(String s) {
    int ans=0;
    int n=s.length();
    for(int i=0;i<n*2-1;i++){
        int left=i/2,right=i/2+i%2;
        while (left>=0&&right<n&&s.charAt(left)==s.charAt(right)){
            left--;
            right++;
            ans++;
        }
    }
    return ans;
}
```

* 剑指 Offer II 021. 删除链表的倒数第 n 个结点

```java
public ListNode removeNthFromEnd(ListNode head, int n) {
    ListNode dummy=new ListNode(-1);
    dummy.next=head;
    ListNode slow=dummy,fast=dummy;
    for(int i=0;i<n;i++){
        fast=fast.next;
    }
    while (fast.next!=null){
        fast=fast.next;
        slow=slow.next;
    }
    slow.next=slow.next.next;
    return dummy.next;
}
```

* 剑指 Offer II 022. 链表中环的入口节点

```java
public ListNode detectCycle(ListNode head) {
    if(head==null)
        return null;
    ListNode slow=head,fast=head;
    while (fast.next!=null&&fast.next.next!=null){
        fast=fast.next.next;
        slow=slow.next;
        if(slow==fast){
            fast=head;
            while (fast!=slow){
                fast=fast.next;
                slow=slow.next;
            }
            return slow;
        }
    }
    return null;
}
```

* 剑指 Offer II 023. 两个链表的第一个重合节点

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode p1=headA;
    while (p1!=null){
        ListNode p2=headB;
        while (p2!=null){
            if(p1==p2)
                return p1;
            p2=p2.next;
        }
        p1=p1.next;
    }
    return null;
}
```

* 剑指 Offer II 024. 反转链表

```java
public ListNode reverseList(ListNode head) {
    ListNode dummy=new ListNode(-1);
    ListNode p=head;
    while (p!=null){
        ListNode ptmp=p.next;
        ListNode tmp=dummy.next;
        dummy.next=p;
        p.next=tmp;
        p=ptmp;
    }
    return dummy.next;
}
```

* 剑指 Offer II 025. 链表中的两数相加

```java
public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    ListNode listNode1=reverse(l1);
    ListNode listNode2=reverse(l2);
    ListNode p1=listNode1,p2=listNode2;
    ListNode dummy=new ListNode(-1);
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
        ListNode p=new ListNode(val);
        ListNode tmp=dummy.next;
        dummy.next=p;
        p.next=tmp;
    }
    return dummy.next;
}
public ListNode reverse(ListNode head){
    ListNode dummy=new ListNode(-1);
    ListNode p=head;
    while (p!=null){
        ListNode tmp=dummy.next;
        dummy.next=p;
        p=p.next;
        dummy.next.next=tmp;
    }
    return dummy.next;
}
```

* 剑指 Offer II 026. 重排链表

```java
public void reorderList(ListNode head) {
    if (head == null) {
        return;
    }
    ListNode mid = middleNode(head);
    ListNode l1 = head;
    ListNode l2 = mid.next;
    mid.next = null;
    l2 = reverseList(l2);
    mergeList(l1, l2);
}

public ListNode middleNode(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;
    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }
    return slow;
}

public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}

public void mergeList(ListNode l1, ListNode l2) {
    ListNode l1_tmp;
    ListNode l2_tmp;
    while (l1 != null && l2 != null) {
        l1_tmp = l1.next;
        l2_tmp = l2.next;

        l1.next = l2;
        l1 = l1_tmp;

        l2.next = l1;
        l2 = l2_tmp;
    }
}
```

* 剑指 Offer II 027. 回文链表

```java
public boolean isPalindrome(ListNode head) {
    ListNode dummy=new ListNode(-1);
    ListNode p=head;
    while (p!=null){
        ListNode tmp=dummy.next;
        dummy.next=new ListNode(p.val);
        p=p.next;
        dummy.next.next=tmp;
    }
    ListNode p1=dummy.next;
    while (head!=null){
        if(p1.val!= head.val)
            return false;
        head=head.next;
        p1=p1.next;
    }
    return true;
}
```

* 剑指 Offer II 028. 展平多级双向链表

```java
private Node cur;
public Node flatten(Node head) {
    if(head==null)
        return null;
    cur=head;
    Node next=head.next;
    Node child=head.child;
    if(child!=null){
        cur.child=null;
        cur.next=child;
        child.prev=cur;
        cur=child;
        flatten(cur);
    }
    if(next!=null){
        cur.next=next;
        next.prev=cur;
        cur=next;
        flatten(cur);
    }
    return head;
}
```

* 剑指 Offer II 032. 有效的变位词

```java
//    public boolean isAnagram(String s, String t) {
//        if(s.equals(t))
//            return false;
//        char[] s1=s.toCharArray();
//        char[] t1=t.toCharArray();
//        Arrays.sort(s1);
//        Arrays.sort(t1);
//        if(s1.length!=t1.length)
//            return false;
//        for(int i=0;i<s1.length;i++){
//            if(s1[i]!=t1[i])
//                return false;
//        }
//        return true;
//    }
    public boolean isAnagram(String s, String t) {
        if(s.equals(t))
            return false;
        char[] s1=s.toCharArray();
        Arrays.sort(s1);
        String s2=new String(s1);
        char[] t1=t.toCharArray();
        Arrays.sort(t1);
        String t2=new String(t1);
        if(s2.equals(t2))
            return true;
        else
            return false;
    }
```

* 剑指 Offer II 033. 变位词组

```java
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String,List<String>> map=new HashMap<>();
    for(String str:strs){
        char[] strtmp=str.toCharArray();
        Arrays.sort(strtmp);
        String key=new String(strtmp);
        List<String> list=map.getOrDefault(key,new ArrayList<String>());
        list.add(str);
        map.put(key,list);
    }
    return new ArrayList<>(map.values());
}
```