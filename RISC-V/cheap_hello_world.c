int main()
{
    char str1[] = "hello,world!\n";
    int sysout = 1;
    int size = 13;
    asm volatile(
        "mv a1 %1 #\n\t"
        "li a0 %0 #\n\t"
        "li a2 %2 #\n\t"
        "li a7 64 #\n\t"
        "ecall    #\n\t"
        :
        : "r"(sysout), "r"(str1), "r"(size)
        : "a0", "a1", "a2", "a7");
    return 0;
}