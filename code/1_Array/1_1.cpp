/*-----------------------------
          二分查找
-----------------------------*/
#include<iostream>
#include<vector>
using namespace std;



class Solution_704{
public:
    int search(vector<int>& nums, int target) {

        int n = nums.size();
        int left = 0, right = n - 1;
        
        while(left <= right){

           int arrow = (left + right) / 2;

           if(nums[arrow] < target){
                left = arrow + 1;
                arrow = (left + right) / 2;
           }

           else if(nums[arrow] > target){
                right = arrow - 1;
                arrow = (left + right) / 2;
           }

           else return arrow;
        }
        
        return -1;
    }
};

class Solution_35 {
public:
    int searchInsert(vector<int>& nums, int target) {
        
        int n = nums.size();
        int left = 0, right = n - 1;
        int arrow = (left + right) / 2;
        
        while(left <= right){

            if(nums[arrow] < target){
                left = arrow + 1;
                arrow = (left + right) / 2;  
            }

            else if(nums[arrow] > target){
                right = arrow - 1;
                arrow = (left + right) / 2;  
            }

            else return arrow;
        }

        if(right < 0) return 0;
        if(nums[arrow] < target) return left;
        else return right;
    }
};

int main()
{
    cout << "------- Q1 -------"<< endl;

    vector<int> nums = {-1,0,3,4,56,102,899,10203};
    int target1 = 899;
    int target2 = 2;

    Solution_704 solution_704; 
    int ans1 = solution_704.search(nums,target1);
    int ans2 = solution_704.search(nums,target2);

    cout << "we find that 899 is no:"<< ans1 << endl;
    cout << "we find that 2 is no:"<< ans2 << endl;
    cout << "-------------------"<< endl;

    cout << "------- Q2 -------"<< endl;

    int target3 = 4;
    int target4 = 10204;

    Solution_35 solution_35; 
    ans1 = solution_35.searchInsert(nums,target1);
    ans2 = solution_35.searchInsert(nums,target2);
    int ans3 = solution_35.searchInsert(nums,target3);
    int ans4 = solution_35.searchInsert(nums,target4);

    cout << "we find that 899 is no:"<< ans1 << endl;
    cout << "we find that 2 is no:"<< ans2 << endl;
    cout << "we find that 4 is no:"<< ans3 << endl;
    cout << "we find that 5 is no:"<< ans4 << endl;
    cout << "-------------------"<< endl;

    return 0;

}