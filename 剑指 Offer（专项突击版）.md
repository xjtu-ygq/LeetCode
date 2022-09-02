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