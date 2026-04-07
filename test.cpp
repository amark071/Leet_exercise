#include<iostream>
#include<vector>
#include<climits>
using namespace std;
int main(){
    int n=2,m=1;
    int d[2][1]={{1},{1}};
    int r[2]={0},c[1]={0};
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            r[i]+=d[i][j];
            c[j]+=d[i][j];
        }
    } 
    int row[3]={0},col[2]={0};
    for(int i=1;i<n+1;i++){
        row[i] = row[i-1]+r[i-1];
    }
    for(int i=1;i<m+1;i++){
        col[i] = col[i-1]+c[i-1];
    }

    int min = INT_MAX;
    for(int i=0;i<n+1;i++){
        if(abs(2*row[i]-row[n+1])<min){
            min = abs(2*row[i]-row[n+1]);
        }
    }
    for(int i=0;i<m+1;i++){
        if(abs(2*col[i]-col[n+1])<min){
            min = abs(2*col[i]-col[n+1]);
        }
    }
    cout << min << endl;
    
    return 0;
}