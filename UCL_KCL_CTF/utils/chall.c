//gcc chall.c -o chall -fno-stack-protector
#include <stdlib.h>
#include<stdio.h>
#include <string.h>

int main(){
    setvbuf(stdout, NULL, _IONBF, 0);
    char vuln [128] = "what is this?";
    char buf [16];
    FILE *fptr;
    fptr = fopen("flag.txt", "r");
    char flag[100];
    fgets(flag, 100, fptr);

    printf("Enter Something: ");
    scanf("%s", buf);

    //This should be impossible to reach!!!
    if(strcmp(vuln, "Unreachable!?!?!?") == 0){
        printf("%s", flag);
    }
    else{
        printf("%s", "Your Input Is: ");
        printf("%s", buf);
        printf("%s", "\n");
    }

    return 0;
}
