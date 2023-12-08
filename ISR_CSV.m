function out = ISR_CSV(w,x,n)

if size(w,2)>1
    w = permute(w,[1 3 2]);
end

signal = mtimesx(w,'C',x-n);
interference = mtimesx(w,'C',n);

out = squeeze(sum(interference.*conj(interference),2)./sum(signal.*conj(signal),2));
