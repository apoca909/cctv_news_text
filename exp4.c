#include "stdio.h"

void main()
{
     int a, b;
     sscanf("123,456", "%d,%d", &a, &b);
     printf("a=%d b=%d\n", a, b);
}