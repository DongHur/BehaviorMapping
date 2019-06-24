% fid = fopen(sprintf('ODE_run%d.tmp', runnum));
fprintf("This is Test Script");
disp("hello");
embeddingValues = zeros(10,1)
for i = 1:10
    fprintf("On line %4i \n", i);
    embeddingValues(i) = i
end

