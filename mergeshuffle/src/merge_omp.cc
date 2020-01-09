#include "fisher_yates.h"
#include <cstdint>

// asm(".align 16              \n\
// perm1: .int 303239696              \n\
//  .int 1803315264              \n\
//  .int 3166732288              \n\
//  .int 3221225472              \n\
// perm2: .int 0              \n\
//  .int 1077952576              \n\
//  .int 2483065856              \n\
//  .int 3918790656              \n\
// mask1: .int 858993459              \n\
//  .int 252645135              \n\
//  .int 16711935              \n\
//  .int 65535              \n\
// mask2: .int 1073741823              \n\
//  .int 54476799              \n\
//  .int 197439              \n\
//  .int 3              \n\
// merge: movq %rdi, %r8              \n\
//  leaq (%rdi,%rsi,4), %rsi              \n\
//  leaq (%rdi,%rdx,4), %r9              \n\
//  movaps perm1, %xmm4              \n\
//  movaps perm2, %xmm5              \n\
//  movaps mask1, %xmm6              \n\
//  movaps mask2, %xmm7              \n\
//  jmp new              \n\
// loop: vmovups (%rdi), %xmm0              \n\
//  vmovups (%rsi), %xmm1              \n\
//  vpslld %xmm2, %xmm4, %xmm3              \n\
//  vpermilps %xmm3, %xmm0, %xmm0              \n\
//  vpslld %xmm2, %xmm5, %xmm3              \n\
//  vpermilps %xmm3, %xmm1, %xmm1              \n\
//  vpsrld %xmm2, %xmm6, %xmm3              \n\
//  vmaskmovps %xmm3, %xmm1, (%rdi)              \n\
//  vpsrld %xmm2, %xmm7, %xmm3              \n\
//  vmaskmovps %xmm3, %xmm0, (%rsi)              \n\
//  movq %r10, %rdi              \n\
//  movq %r11, %rsi              \n\
//  shrq $4, %rax              \n\
//  subl $4, %ecx              \n\
//  jnz skip              \n\
// new: rdrand %rax              \n\
//  jnc new              \n\
//  movl $64, %ecx              \n\
// skip: movl %eax, %edx              \n\
//  andl $15, %edx              \n\
//  addl %edx, %edx              \n\
//  vmovd %edx, %xmm2              \n\
//  popcntl %edx, %edx              \n\
//  leaq 16(%rdi), %r10              \n\
//  leaq (%rsi,%rdx,4), %r11              \n\
//  cmpq %rsi, %r10              \n\
//  ja skip2              \n\
//  cmpq %r9, %r11              \n\
//  jna loop              \n\
//  jmp skip2              \n\
// swap: cmpq %r9, %rsi              \n\
//  jnb out2              \n\
//  movl (%rdi), %r10d              \n\
//  movl (%rsi), %r11d              \n\
//  movl %r11d, (%rdi)              \n\
//  movl %r10d, (%rsi)              \n\
//  addq $4, %rsi              \n\
// test: addq $4, %rdi              \n\
//  shrq %rax              \n\
//  decl %ecx              \n\
//  jnz skip2              \n\
// new2: rdrand %rax              \n\
//  jnc new2              \n\
//  movl $64, %ecx              \n\
// skip2: testb $1, %al              \n\
//  jnz swap              \n\
//  cmpq %rsi, %rdi              \n\
//  jb test              \n\
// out2: movq %r9, %rax              \n\
//  subq %r8, %rax              \n\
//  shrq $2, %rax              \n\
//  bsrl %eax, %ecx              \n\
//  xorl %eax, %eax              \n\
//  btsl %ecx, %eax              \n\
//  leal -1(%rax,%rax), %ecx              \n\
//  jmp start3              \n\
// new3: rdrand %eax              \n\
//  jnc new3              \n\
//  andl %ecx, %eax              \n\
//  leaq (%r8,%rax,4), %rsi              \n\
//  cmpq %rdi, %rsi              \n\
//  ja new3              \n\
//  movl (%rdi), %eax              \n\
//  movl (%rsi), %edx              \n\
//  movl %edx, (%rdi)              \n\
//  movl %eax, (%rsi)              \n\
//  addq $4, %rdi              \n\
// start3: cmpq %r9, %rdi              \n\
//  jb new3              \n\
//  ret");

/* extern "C" */ void merge( unsigned int* t, unsigned int m, unsigned int n );

void parallel_merge_shuffle( unsigned int* t, uint64_t n )
{
    unsigned int c = 0;
    while( ( n >> c ) > cutoff )
        c++;
    unsigned int q = 1 << c;
    unsigned long nn = n;

#pragma omp parallel for
    for( unsigned int i = 0; i < q; i++ )
    {
        unsigned long j = nn * i >> c;
        unsigned long k = nn * ( i + 1 ) >> c;
        fisher_yates( t + j, k - j );
    }

    for( unsigned int p = 1; p < q; p += p )
    {
#pragma omp parallel for
        for( unsigned int i = 0; i < q; i += 2 * p )
        {
            unsigned long j = nn * i >> c;
            unsigned long k = nn * ( i + p ) >> c;
            unsigned long l = nn * ( i + 2 * p ) >> c;
            merge( t + j, k - j, l - j );
        }
    }
}
