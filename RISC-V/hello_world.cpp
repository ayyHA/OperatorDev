int read(int __fd, const void *__buf, int __n)
{
    int bytes;
    __asm__ __volatile__(
        "mv a0, %1           # file descriptor\n"
        "mv a1, %2           # buffer \n"
        "mv a2, %3           # size \n"
        "li a7, 63           # syscall read (63) \n"
        "ecall \n"
        "mv %0, a0"
        : "=r"(bytes)                     // Output list
        : "r"(__fd), "r"(__buf), "r"(__n) // Input list
        : "a0", "a1", "a2", "a7");
    return bytes;
}

void write(int __fd, const void *__buf, int __n)
{
    __asm__ __volatile__(
        "mv a0, %0           # file descriptor\n"
        "mv a1, %1           # buffer \n"
        "mv a2, %2           # size \n"
        "li a7, 64           # syscall write (64) \n"
        "ecall"
        :                                 // Output list
        : "r"(__fd), "r"(__buf), "r"(__n) // Input list
        : "a0", "a1", "a2", "a7");
}

int main()
{
    char str[11];
    int n = read(0, str, 11);
    printf("We already read %d chars", n);
    write(1, str, n);
    return 0;
}

void _start()
{
    main();
}