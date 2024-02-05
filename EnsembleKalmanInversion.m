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

%% Getting directory of HVSR data derived using HVSR code (freq, hvsr-mean, hvsr-std)
d = dir('C:\Users\user\Desktop\Thesis\VsProfiles\Final_Vs_Profiles\csv_Jian\*.txt'); %reading all Vs files with txt format
mkdir result      % creating a directory to save the results for each station

%% Create optimization options
options = optimoptions('quadprog','Display','off');

%% Start Performing HVSR Inversion
for ss = 1:size(d,1) 
    clearvars -except ss d options 
    
    data_Vs = readtable(['C:\Users\user\Desktop\Thesis\VsProfiles\Final_Vs_Profiles\csv_Jian\',d(ss).name]);  %reading Vs profile
    Vsreal = data_Vs.Vs;       %Vs values in experimental Vs profile
    Depthreal = data_Vs.Depth;  %Depth values in experimental Vs profile

    Depth_new=0:0.5:Depthreal(end); Depth_new = Depth_new';
%     Vsreal_modified = interp1(Depthreal,Vsreal,Depth_new);
    for jjj = 1:length(Depth_new)-1
        for iii = 2:length(Depthreal)
            if Depthreal(iii-1,1)<=Depth_new(jjj) && Depth_new(jjj)<Depthreal(iii)
                Vsreal_modified(jjj+1,1) = Vsreal(iii,1);
            end 
        end
    end
    Vsreal_modified(1)=Vsreal(1);
    Vsreal_modified = [Vsreal_modified;Vsreal(end)];
    Vsreal_modified = interp1(Depthreal,Vsreal,Depth_new); 
    

    Rfac   = 0.05*Vsreal_modified;  % noise created based on real values to ba added to measured values (beta * y)
    Gamma  = diag(Rfac).^2;   % diagonal matrix of noise values (eq. 2 of paper)

    nb = 3; % number of parametres [Vs0, n, k]
    nParticle = 100; % creating intial ensemble of N particles (page 3)

    % defining A as coefficient matrix that should satisfy Au<g (g includes defined constraints)
    Auu = [1 zeros(1,2);-1 zeros(1,2);0 1 0;0 -1 0;0 0 1;0 0 -1];
    
    gu = [500/100, -10/100, 100,0,1,0]'; % upperbound values vector
    
    vs0_mean = Vsreal_modified(1); vs0_lb=vs0_mean-100; vs0_ub = vs0_mean+100;
    vs0 = (((vs0_ub-vs0_lb)*rand(nParticle,1))/100)+vs0_lb/100;
    k_mean= (Vsreal_modified(end)-Vsreal_modified(1))/(Depth_new(end)-Depth_new(1)); k_lb=k_mean-0.5; k_up=k_mean+0.5;
    k = (k_up-k_lb)*rand(nParticle,1)+k_lb;
    %vs0= (490.*rand(nParticle,1)+10)/100; %normalization;
    %k = 10.*rand(nParticle,1);
    n =0.6*rand(nParticle,1)+0.2;
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
    iteration = 0; maxiter = 100;
    uhattemp = uhat;
    s = 0; 
    Gammau = diag((0.01*uvec).^2); %considering a diagonal noise matrix on uvec
    mypath0 = [pwd,'/'];
    dir='Est'; mypath=[mypath0,dir]; mkdir(mypath);
    fig1 = figure(1); fig2 = figure(2); fig3 = figure(3); 
    
    while iteration <  maxiter
        iteration = iteration+1;
        %% forward model
        for ii = 1:nParticle
            for i =1:length(Depth_new)
                if Depth_new(i)<2.5
                    ParticleVs(i,ii) = uhat(1,ii)*100;
                else
                    ParticleVs(i,ii) = uhat(1,ii)*100*(1+uhat(2,ii)*((Depth_new(i)-2.5))^(uhat(3,ii)));
                end
            end
        end
        
        what = ParticleVs;
        uhatold = uhat;
        
        wnp1 = mean(what,2); %calculating mean of Vs of all particles 
        unp1 = mean(uhat,2); %calculating mean of uhat of all particles 
        Cuwnp1 = zeros(nb,length(Vsreal_modified)); %covariance matrix of uhat and what-size:(3,12)
        Cwwnp1 = zeros(length(Vsreal_modified)); %covariance matrix of what-size:(12,12)
        for i = 1:nParticle
            Cuwnp1   = Cuwnp1   + (uhat(:,i)-unp1)*(what(:,i)-wnp1)'/(nParticle-1);
            Cwwnp1   = Cwwnp1   + (what(:,i)-wnp1)*(what(:,i)-wnp1)'/(nParticle-1);
        end
        
        Ik = eye(size(Cwwnp1)); %Identity matrix-size 500*500 
        K = Cuwnp1*((Cwwnp1+Gamma)\Ik); % Adding noise to Cww but then?????
        
        temp = mvnrnd(zeros(length(Vsreal_modified),1),Gamma,nParticle); % returns a matrix "temp" of "nParticle"
        % random vectors chosen from the same multivariate normal distribution, 
        % with mean vector  "zeros" and noise matrix Gamma
        
        yperturb = temp'; 
        
        for i = 1:nParticle
            dy = Vsreal_modified-what(:,i)+s*yperturb(:,i); %calculating the difference between experimental and theoritical
            % Vs and adding noise to them
            uhattemp(:,i) = uhat(:,i) + K*dy; % ???? which part of the paper? K is kalman gain Eq 3 in the second paper
        end

        CC = zeros(size(Auu,1) ,nParticle); %6*18
        Bu = zeros(size(uhat,1),nParticle); %3*18
        Bw = zeros(length(Vsreal_modified),nParticle); %12*18
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
                f    =  1/nParticle*Bw'*(Gamma\(what(:,i)-Vsreal_modified-s*yperturb(:,i)));
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

        %% Check final optimized values for vs0, k, n

        for ii = 1:nParticle
            for i =1:length(Depth_new)
                if Depth_new(i)<2.5
                    ParticleVs(i,ii) = uhat(1,ii)*100;
                else
                    ParticleVs(i,ii) = uhat(1,ii)*100*(1+uhat(2,ii)*((Depth_new(i)-2.5))^(uhat(3,ii)));
                end
            end
        end
        unp_final = mean(uhat, 2);
        unp_final(1) = unp_final(1)*100;
        for i =1:length(Depth_new)
            if Depth_new(i)<2.5
                ParticleVs2(i) = unp_final(1);
            else
                ParticleVs2(i) = unp_final(1)*(1+unp_final(2)*((Depth_new(i)-2.5))^(unp_final(3)));
            end
        end
       
        mean_ParticleVs = mean(ParticleVs,2);
        % Plotting profiles of each particle and compare with the mean and measured profiles

        %% Plotting Profile
        set(0,'CurrentFigure',fig2);clf;hold all; 
        for i=1:nParticle
            b1 = stairs(ParticleVs(:,i),Depth_new,'color',[0.65 0.65 0.65]); set (gca,'Ydir','reverse')%, 'XLim',([0 1000]) );
        end
        b2=stairs(ParticleVs2(:,:),Depth_new,'r','linewidth',2);set (gca,'Ydir','reverse')%,'XLim',([0 1000]));
        b3=stairs(mean_ParticleVs,Depth_new,'k','linewidth',2);set (gca,'Ydir','reverse')%,'XLim',([0 1000]));
        b4 = stairs(Vsreal_modified,Depth_new,'b','linewidth',2);set (gca,'Ydir','reverse')%, 'XLim',([0 1000]));  xlim=([0 2500]); 
        xlabel('Vs'); ylabel('Depth');title(['Vs Profile - Iteration:',num2str(iteration)]);
        legend([b1 b2 b3 b4],{'Particles','Mean of uhat', 'Mean of Vs Profiles','Measured'},'Location','northeast','Orientation','vertical'); %drawnow; 
        saveas(gca, ["result_plot/Profile _ "+num2str(iteration)+".png"]);
        %% Plotting Convergency
        set(0,'CurrentFigure',fig3);clf;hold all;
        for i =1:nb
            plot(1:iteration+1,uvec(i,:),'r-x');grid on;
        end
        saveas(gca, ["result_plot/iteration.png"]);
        %% Error Value
        Vs1 = ParticleVs2';
        eps_EK = (sum(abs(log(Vsreal_modified)-log(Vs1))))/length(Depth_new);

    end

    mkdir(['result/',d(ss).name]);
    csvwrite(['result/',d(ss).name,'/Vs0_k_n.txt'],unp_final);
    csvwrite(['result/',d(ss).name,'/Vs.txt'],[Depth_new, Vsreal_modified, ParticleVs2']);
    csvwrite(['result/',d(ss).name,'/Vs_measured.txt'],[Depthreal, Vsreal]);
    csvwrite(['result/',d(ss).name,'/uhat.txt'],uhat);
end
