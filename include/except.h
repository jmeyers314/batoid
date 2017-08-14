#ifndef batoid_except_h
#define batoid_except_h

namespace batoid {

    class NoIntersectionError
    {
    public:
        explicit NoIntersectionError(const char* _message) : message(_message) {}
        const char* GetMessage() const {return message;}
    private:
        const char* const message;
    };

    class NoFutureIntersectionError
    {
    public:
        explicit NoFutureIntersectionError(const char* _message) : message(_message) {}
        const char* GetMessage() const {return message;}
    private:
        const char* const message;
    };

    class NotImplemented : public std::logic_error {
    public:
        NotImplemented(std::string s="") : std::logic_error(s) {}
    };

}

#endif
