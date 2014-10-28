function savePlot(xLabel, yLabel, filename)
% Helper function provided by Emti
% Save the latest figure to PDF with filename `name`
    hx = xlabel(xLabel);
    hy = ylabel(yLabel);

    % Makes the plot look nice (font size, etc)
    set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.02 .02],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
    set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
    grid on;

    % Print the file to pdf
    print('-dpdf', [filename, '.pdf']);

    % Next you should CROP PDF using pdfcrop in linux and mac
end