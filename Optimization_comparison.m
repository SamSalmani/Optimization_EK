clc; clear; close all;

%% Creating parallel computing toolbox to use 10 cores
poolobj = gcp('nocreate');
seedchange = 'No';
nparpro=10;
if ~isempty(poolobj) && strcmp(seedchange,'Yes')
    delete(poolobj);
    parpool(nparpro);
elseif isempty(poolobj)
    parpool(nparpro);
end

%% Start
d = dir('C:\Users\user\Desktop\Thesis\VsProfiles\Final_Vs_Profiles\csv_Jian\*.txt'); %reading all Vs files with txt format
mkdir result      % creating a directory to save the results for each station
options = optimoptions('quadprog','Display','off');

for ss = 1:size(dirlist,1) 
    clearvars -except ss d options 
    
    data_Vs = readtable(['C:\Users\user\Desktop\Thesis\VsProfiles\Final_Vs_Profiles\csv_Jian\',d(ss).name]);  %reading Vs profile
    Vsreal = data_Vs.Vs;       %Vs values in experimental Vs profile
    Depthreal = data_Vs.Depth;  %Depth values in experimental Vs profile

    Rfac   = 0.2*Vsreal;  % noise created based on real values to ba added to measured values (beta * y)
    Gamma  = diag(Rfac).^2;   % diagonal matrix of noise values (eq. 2 of paper)

    nb = 3; % number of parametres [Vs0, n, k]
    nParticle = 100; % creating intial ensemble of N particles (page 3)

    % defining A as coefficient matrix that should satisfy Au<g (g includes defined constraints)
    Auu = [1 zeros(1,2);-1 zeros(1,2);0 1 0;0 -1 0;0 0 1;0 0 -1];

    gu = [500/100, -100/100, 10,-0.1,5,-1]'; % upperbound values vector

    vs0= (400.*rand(nParticle,1)+100)/100; %normalization;
    k = 5.*rand(nParticle,1);
    n =4.*rand(nParticle,1)+1;
    uhat = [vs0, k, n]'; 
    
    count = 0;
    for j = 1:nParticle 
        theta = uhat(:,j); %reading the values of Vs0,k and n 
        V = find(Auu*theta>gu); %finding the index of rows which violates the rule : Au<g
        if isempty(V) %if empty,nothing happens
        else % use quadprog optimization to update uhat to satisfy the constraint
            count = count+1;
            M    =  eye(nb);
            f    =  -theta;
            uhat(:,j) = quadprog(M,f,Auu,gu,[],[],[],[],[],options);
        end
    end

    uvec(:,1) = mean(uhat,2); %calculating the mean of all columns of uhat
    iteration = 0; maxiter = 150;
    uhattemp = uhat;
    s = 1; 
    Gammau = diag((0.05*uvec).^2); %considering a diagonal noise matrix on uvec

    %% While Loop
    fig2 = figure(2);
    while iteration <  maxiter
        iteration = iteration+1;
        % forward model
        for ii = 1:nParticle
            for i =1:length(Depthreal)
                if Depthreal(i)<2.5
                    ParticleVs(i,ii) = uhat(1,ii)*100;
                else
                    ParticleVs(i,ii) = uhat(1,ii)*100*(1+uhat(2,ii)*((Depthreal(i)-2.5))^(1/uhat(3,ii)));
                end
            end
        end
        
        what = ParticleVs;
        uhatold = uhat;
        
        wnp1 = mean(what,2); %calculating mean of Vs of all particles 
        unp1 = mean(uhat,2); %calculating mean of uhat of all particles 
        Cuwnp1 = zeros(nb,length(Vsreal)); %covariance matrix of uhat and what-size:(3,12)
        Cwwnp1 = zeros(length(Vsreal)); %covariance matrix of what-size:(12,12)
        for i = 1:nParticle
            Cuwnp1   = Cuwnp1   + (uhat(:,i)-unp1)*(what(:,i)-wnp1)'/(nParticle-1);
            Cwwnp1   = Cwwnp1   + (what(:,i)-wnp1)*(what(:,i)-wnp1)'/(nParticle-1);
        end
        
        Ik = eye(size(Cwwnp1)); %Identity matrix-size 500*500 
        K = Cuwnp1*((Cwwnp1+Gamma)\Ik); % Adding noise to Cww but then?????
        
        temp = mvnrnd(zeros(length(Vsreal),1),Gamma,nParticle);
        yperturb = temp'; 
        
        for i = 1:nParticle
            dy = Vsreal-what(:,i)+s*yperturb(:,i); 
            uhattemp(:,i) = uhat(:,i) + K*dy; 
        end

        CC = zeros(size(Auu,1) ,nParticle); %6*18
        Bu = zeros(size(uhat,1),nParticle); %3*18
        Bw = zeros(length(Vsreal),nParticle); %12*18
        for i = 1:nParticle
            Bu(:,i) = uhat(:,i) - unp1;
            Bw(:,i) = what(:,i) - wnp1;
        end
        for i = 1:nParticle
            V = find(Auu*uhattemp(:,i)>gu);
            if isempty(V)
                uhat(:,i) = uhattemp(:,i);
            else
                M    =  1/nParticle/nParticle*Bw'*(Gamma\Bw)+eye(nParticle)/nParticle;
                f    =  1/nParticle*Bw'*(Gamma\(what(:,i)-Vsreal-s*yperturb(:,i)));
                A    = Auu*Bu/nParticle;
                ub   = gu-Auu*uhat(:,i);
                bhat = quadprog(M,f,A,ub,[],[],[],[],[],options);
                uhat(:,i) = uhat(:,i)+1/nParticle*Bu*bhat;
            end
        end
        
        unp1   = mean(uhat,2); 
        uvec(:,iteration+1) = unp1;
        
        temp = mvnrnd(zeros(nb,1),Gammau,nParticle);
        uperturb = temp';
        
        if iteration < maxiter 
            uhat = uhat+uperturb;
            for j = 1:nParticle
                while ~isempty(find(Auu*uhat(:,j)>gu,1))
                    M     =  eye(nb);
                    f     =  -uhat(:,j);
                    uhat(:,j) = quadprog(M,f,Auu,gu,[],[],[],[],[],options);
                end
            end
        end
        uhatplot = uhat;
   
        % Check final optimized values for vs0, k, n
        Depth_new=0:0.1:Depthreal(end); Depth_new = Depth_new';
        Vsreal_modified = interp1(Depthreal(1:end-1),Vsreal(1:end-1),Depth_new); 
        unp_final = mean(uhat, 2);
        unp_final(1) = unp_final(1)*100;
        for i =1:length(Depth_new)
            if Depth_new(i)<2.5
                ParticleVs2(i) = unp_final(1);
            else
                ParticleVs2(i) = (unp_final(1)*(1+unp_final(2)*((Depth_new(i)-2.5))^(1/unp_final(3))));
            end
        end
        mean_ParticleVs = mean(ParticleVs,2);
        
        set(0,'CurrentFigure',fig2);clf;hold all; 
        for i=1:nParticle
            b1 = stairs(ParticleVs(:,i),Depthreal,'color',[0.65 0.65 0.65]); set (gca,'Ydir','reverse', 'XLim',([0 1000]) );
        end
        b2=plot(ParticleVs2(:,:),Depth_new,'r','linewidth',2);set (gca,'Ydir','reverse','XLim',([0 1000]));
        b3=stairs(mean_ParticleVs,Depthreal,'k','linewidth',2);set (gca,'Ydir','reverse','XLim',([0 1000]));
        b4 = stairs(Vsreal,Depthreal,'b','linewidth',2);set (gca,'Ydir','reverse', 'XLim',([0 1000]));  xlabel('Vs'); ylabel('Depth');title(['Vs Profile - Iteration:',num2str(iteration)])  
    end

    %%  Optimization
    %%%%% Using the minimum error function 
    epsilonvs=[];
    Depth_new=0:0.1:Depthreal(end); Depth_new = Depth_new';
    Vsreal_modified = interp1(Depthreal(1:end-1),Vsreal(1:end-1),Depth_new); 
    for j=1:length(vs0)
        N=length(Depth_new);
        x0=uhat(:,j);
        for i=1:N  
            if Depth_new(i)<=2.5
                objective1 = uhat(1,j);
            else
                objective = @(x) x(1)*(1+x(2)*(Depth_new(i)-2.5))^(1/x(3));
                objective1 = abs(objective(x0));
            end
            func = abs(log(objective1)-log(Vsreal_modified(i)));
            epsilonvs = [epsilonvs; func];
        end 
    end
    epsilonvs = reshape(epsilonvs,[N,length(vs0)]);
    
    epsilonvs_total=[];
    for j = 1:size(epsilonvs,2)
        epsilonvs_total = [epsilonvs_total,sum(epsilonvs(:,j))/N];
    end
    index = find(epsilonvs_total == min(epsilonvs_total));
    uhat_final = uhat(:,index); 
    
    Vs_meas = [];
    for i=1:length(Depth_new)
        if Depth_new(i)<=2.5
            objective1 = uhat_final(1)*100;
        else 
            objective = @(a) a(1)*100*(1+a(2)*(Depth_new(i)-2.5))^(1/a(3));
            objective1 = objective(uhat_final);
        end
        Vs_meas = [Vs_meas;abs(objective1)];
    end 
    Profile1 = [Depth_new,Vsreal_modified,Vs_meas];
    set(0,'CurrentFigure',fig2);hold all; 
    b5 = plot(Vs_meas,Depth_new,'g', 'linewidth',2); set (gca,'Ydir','reverse', 'XLim',([0 500]) );
   
    %%%%%%%%%%%%%%% fmincon
    options1 = optimset('PlotFcns','optimplotfval','TolFun',1e-8, 'MaxFunEvals', 2000);
    x0=[2,2,2];
    func = @(x) Epsilon2(x,x0, Vsreal, Depthreal);
    func2 = @(x) Epsilon(x,x0, Vsreal_modified, Depth_new);
    A = [1 zeros(1,2);-1 zeros(1,2);0 1 0;0 -1 0;0 0 1;0 0 -1];
    b = [500/100, -100/100, 10,-0.1,5,-1]';
    [xfmincon, fval1] = fmincon(func, x0,A,b,[],[],[],[],[],options1);
    [xfminsearch, fval2] = fminsearch(func2, x0,options1);

    Depth_new=0:0.1:Depthreal(end);
    Vs_meas_fmincon = [];
    for i=1:length(Depth_new)
        if Depth_new(i)<=2.5
            objective1 = xfmincon(1)*100;
        else 
            objective = @(x) x(1)*100*(1+x(2)*(Depth_new(i)-2.5))^(1/x(3));
            objective1 = objective(xfmincon);
        end
        Vs_meas_fmincon = [Vs_meas_fmincon;abs(objective1)];
    end 
    Profile2 = [Depth_new',Vsreal_modified,Vs_meas_fmincon];
    set(0,'CurrentFigure',fig2);hold all; 
    b6 = plot(Vs_meas_fmincon,Depth_new,"LineStyle","--","Color",'m', 'linewidth',2); set (gca,'Ydir','reverse', 'XLim',([0 500]) );
   
    %%%%%% fimnsearch
    Depth_new=0:0.1:Depthreal(end);
    Vs_meas_fminsearch = [];
    for i=1:length(Depth_new)
        if Depth_new(i)<=2.5
            objective1 = xfminsearch(1)*100;
        else 
            objective = @(x) x(1)*100*(1+x(2)*(Depth_new(i)-2.5))^(1/x(3));
            objective1 = objective(xfminsearch);
        end
        Vs_meas_fminsearch = [Vs_meas_fminsearch;abs(objective1)];
    end 
    Profile4 = [Depth_new',Vsreal_modified,Vs_meas_fminsearch];
    set(0,'CurrentFigure',fig2);hold all; 
    b8 = plot(Vs_meas_fminsearch,Depth_new,"Color",'c', 'linewidth',2); set (gca,'Ydir','reverse', 'XLim',([0 500]) );
    
    
    xlabel('Vs'); ylabel('Depth');title(["Profile-"+d(ss).name(9:end-4)+'-Iteration:'+num2str(iteration)]);
    legend([b1 b2 b3 b4 b5 b6 b8],{'Particles','Mean of uhat', 'Mean of Vs Profiles','Measured', "Minimizing Error function", "Fmincon", "Fminsearch"},'Location','northeast','Orientation','vertical'); 
    mkdir(['result_plot/', d(ss).name]); 
    saveas(gca, ["result_plot/"+ d(ss).name+ "/Profile _ "+num2str(iteration)+".png"]);

    %% Calculating error
    Vs1 = ParticleVs2';
    Vs2 = Vs_meas;
    Vs3 = Vs_meas_fmincon;
    Vs4 = Vs_meas_fminsearch;
    eps_EK = (sum(abs(log(Vsreal_modified)-log(Vs1))))/length(Depth_new);
    eps_minimizing_error = (sum(abs(log(Vsreal_modified)-log(Vs2))))/length(Depth_new);
    eps_fmincon = (sum(abs(log(Vsreal_modified)-log(Vs3))))/length(Depth_new);
    eps_fminsearch = (sum(abs(log(Vsreal_modified)-log(Vs4))))/length(Depth_new);
end
