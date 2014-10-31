function prettifyPlot(xLabel, yLabel)
    if(nargin < 3)
        yLabel = '';
    end;
    if(nargin < 2)
        xLabel = '';
    end;
    
    hx = xlabel(xLabel);
    hy = ylabel(yLabel);

    % Makes the plot look nice (font size, etc)
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;
end