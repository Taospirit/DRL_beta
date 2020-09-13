#include <iostream>
class ClassA
{
    public:
        void f1(){std::cout.put('a');}
        virtual void f2(){std::cout.put('b');}
        void f3(){f1();}
        void f4(){f1();}
        void f5(){f2();}
        virtual void f6(){f4();}
};
class ClassB: public ClassA
{
    public:
    void f1(){std::cout.put('A');}
    virtual void f2(){std::cout.put('B');}
    void f4(){f1();}
    void F(){f1();f2();f3();f4();f5();f6();}
};
int main()
{
    ClassB B;
    B.F();
    return 0;
}
