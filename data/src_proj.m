addpath('~fessler/l/web/irt/irt/'), setup

load('results_after.mat')
figure, im([squeeze(atvol(:,:,:,112));squeeze(vol(:,:,:,112));warpvol(:,:,112)])
title '', axis off
set(gca,'fontsize',24)

figure, im([atseg(:,:,112);squeeze(seg(:,:,:,112));warpseg(:,:,112)])
title('Fix                  Move            Mapped')
set(gca,'fontsize',28)

