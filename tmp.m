function tmp()

if 2>1
    a = 1;
end

t();
a


function t()
global a
disp(a)