#include "stdio.h"

void exp4_task1()
{
     // 输入两个正整数，求其中最大公约数和最小公倍数。
     int a, b, n, m;
     sscanf("32 24", "%d %d", &a, &b);
     printf("a=%d b=%d\n", a, b);
     m = a * b;
     n = a % b;
     while (n != 0)
     {
          a = b;
          b = n;
          n = a % b;
     }
     n = b;
     printf("%d %d\n", n, m / n);
}
void main()
{
     exp4_task1();
}