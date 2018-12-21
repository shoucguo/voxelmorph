cd ./ir74

%%
load results1_before.mat
load results1_failed.mat

atvol = squeeze(atvol);
vol = squeeze(vol);
pred = squeeze(pred);

%%
% pred
pred = pred/max(abs(pred(:)));
pred = pred-min(pred(:))/(max(pred(:)-min(pred(:))));

figure, imshow(permute(squeeze(pred(:,:,112,:)),[2,1,3]))
figure, imshow(squeeze(pred(90,:,:,:)))
figure, imshow(permute(squeeze(pred(:,54,:,:)),[2,1,3]))

% atvol, vol, warpvol
figure, im([squeeze(atvol(:,:,112));squeeze(vol(:,:,112));squeeze(warpvol(:,:,112))])
axis off, title ''
figure, im([squeeze(atvol(90,:,:))';squeeze(vol(90,:,:))';squeeze(warpvol(90,:,:))'])
axis off, title ''
figure, im([squeeze(atvol(:,64,:));squeeze(vol(:,64,:));squeeze(warpvol(:,64,:))])
axis off, title ''

%%
load results_after.mat 
atvol = squeeze(atvol);
vol = squeeze(vol);
seg = squeeze(seg);
pred = squeeze(pred);

% pred
pred = pred/max(abs(pred(:)));
pred = pred-min(pred(:))/(max(pred(:)-min(pred(:))));

figure, imshow(permute(squeeze(pred(:,:,112,:)),[2,1,3]))
figure, imshow(squeeze(pred(90,:,:,:)))
figure, imshow(flip(permute(squeeze(pred(:,44,:,:)),[2,1,3]),1))

% atvol, vol, warpvol
figure, im([[squeeze(atvol(:,:,146)),squeeze(atvol(:,:,100)),squeeze(atvol(:,:,76))];...
    [squeeze(vol(:,:,146)),squeeze(vol(:,:,100)),squeeze(vol(:,:,76))]; ...
    [squeeze(warpvol(:,:,146)),squeeze(warpvol(:,:,100)),squeeze(warpvol(:,:,76))]])
axis off, title ''

figure, im([squeeze(atvol(:,:,112));squeeze(vol(:,:,112));squeeze(warpvol(:,:,112))])
axis off, title ''
figure, im([squeeze(atvol(90,:,:))';squeeze(vol(90,:,:))';squeeze(warpvol(90,:,:))'])
axis off, title ''
figure, im([flip(squeeze(atvol(:,44,:)),2);flip(squeeze(vol(:,44,:)),2);flip(squeeze(warpvol(:,44,:)),2)])
axis off, title ''

% atseg, seg, warpseg
figure, im([squeeze(atseg(:,:,112));squeeze(seg(:,:,112));squeeze(warpseg(:,:,112))])
axis off, title ''
figure, im([squeeze(atseg(90,:,:))';squeeze(seg(90,:,:))';squeeze(warpseg(90,:,:))'])
axis off, title ''
figure, im([flip(squeeze(atseg(:,44,:)),2);flip(squeeze(seg(:,44,:)),2);flip(squeeze(warpseg(:,44,:)),2)])
axis off, title ''

