function [sample, trueTangentSpace, parametrization] = generateSampleOnSurface(pointsNumber, surfaceString)
  sample = zeros(pointsNumber, 3);
  trueTangentSpace = cell(pointsNumber, 1);
  switch surfaceString
    case 'saddle'
      x = @(u,v) u;
      y = @(u,v) v;
      z = @(u,v) u.^2-v.^2;

      G = @(u,v) sqrt(4*u.^2+4*v.^2+1);
      q = @(u,v) 1/4;
      S = dblquad(G, -1, 1, -1, 1);
      p = @(u,v) 1/S*G(u,v);
      M = 12/S;
      alreadyGenerated = 1;
      p0 = [];
      p1 = [];
      while alreadyGenerated <= pointsNumber
          u = rand(1)*2 - 1;
          v = rand(1)*2 - 1;
          testProbability = rand(1);
          if testProbability <= p(u,v)/(M*q(u,v))
              p0 = [p0; u];
              p1 = [p1; v];
              sample(alreadyGenerated,:) = [x(u,v) y(u,v) z(u,v)];
              [trueTangentSpace{alreadyGenerated}, ~] = qr([1, 0, 2 * u; 0, 1, -2 * v]', 0);
              alreadyGenerated = alreadyGenerated+1;
          end
      end
      u = p0;
      v = p1;
    case 'ellipsoid'
      u=0:0.1:pi/2;
      v=0:0.1:2*pi;
      size(u);
      size(v);
      u=[u 0];
      v=[v 0];
      [u v]=meshgrid(u,v);

      SX=11.9045; %Poschitali v wolframalpha

      G=sqrt( (cos(u).^2.*cos(v).^2+cos(u).^2+1).*(sin(u).^2.*sin(v).^2+2.*sin(u).^2) -(cos(u).*sin(u).*cos(v).*sin(v)).^2 );
      size(G);

      p=1/SX*G;
      q=1/(pi/2*2*pi);
      max_p=max(max(p));

      M=max_p/q;

      s=[];
      k=1;%proverka uslovii dli while
      while(k==1)
          u0=rand()*pi/2;
          v0=rand()*2*pi;

          t=func_p(u0,v0)/M/q;
          if (t > 1)
            disp(t);
          end
          if ( rand()<=t)
              s=[s; u0 v0];
          end
          if (size(s,1)==pointsNumber)
              k=0;
          end;
      end

      u=s(:,1);
      v=s(:,2);
      for i=1:pointsNumber
          sample(i, :) = [sqrt(3)*sin(u(i)).*cos(v(i)), sqrt(2)*sin(u(i)).*sin(v(i)), cos(u(i))];
          [trueTangentSpace{i}, ~] = qr([sqrt(3)*cos(u(i)).*cos(v(i)), sqrt(2)*cos(u(i)).*sin(v(i)), -sin(u(i)); ...
            -sqrt(3)*sin(u(i)).*sin(v(i)), sqrt(2)*sin(u(i)).*cos(v(i)), 0]', 0);
      end
    case 'cylinder'
      % randMatrix = orth(rand(2, 2)); %
      sample = zeros(pointsNumber,3);
      k = 1;
      M = 2*pi*sqrt(3)/9.9;
      f=@(fi,h) sqrt(2*(cos(fi)).^2+3*(sin(fi)).^2);
      c=2*pi;
      u = [];
      v = [];
      while k <= pointsNumber
          x0 = [rand(1)*pi,rand(1)*2];
          test = [rand(1)*pi,rand(1)*2];
          if test <= f(x0(1), x0(2))/M/c;
              u = [u; x0(1)];
              v = [v; x0(2)];
              sample(k,1) = sqrt(3) * cos(x0(1));
              sample(k,2) = sqrt(2) * sin(x0(1));
              sample(k,3)=x0(2);
              sample(k, :) = sample(k, :);
              [trueTangentSpace{k}, ~] = qr([-sqrt(3) * sin(x0(1)), sqrt(2) * cos(x0(1)), 0; 0, 0, 1]', 0);
              trueTangentSpace{k} = trueTangentSpace{k}; % * randMatrix; %
              k = k+1;
          end
      end
    case 'sphere'
      u=0:0.1:pi/2;
      v=0:0.1:2*pi;
      size(u);
      size(v);
      u=[u 0];
      v=[v 0];
      [u v]=meshgrid(u,v);

      SX = 2 * pi; %Poschitali v wolframalpha

      G=sqrt( (cos(u).^2.*cos(v).^2+cos(u).^2+1).*(sin(u).^2.*sin(v).^2+sin(u).^2) -(cos(u).*sin(u).*cos(v).*sin(v)).^2 );
      size(G);

      p=1/SX*G;
      q=1/(pi/2*2*pi);
      max_p=max(max(p));

      M=max_p/q;

      s=[];
      k=1;%proverka uslovii dli while
      while(k==1)
          u0=rand()*pi/2;
          v0=rand()*2*pi;

          t=func_p(u0,v0)/M/q;
          if (t > 1)
            disp(t);
          end
          if ( rand()<=t)
              s=[s; u0 v0];
          end
          if (size(s,1)==pointsNumber)
              k=0;
          end;
      end

      u=s(:,1);
      v=s(:,2);
      for i=1:pointsNumber
          sample(i, :) = [sin(u(i)).*cos(v(i)), sin(u(i)).*sin(v(i)), cos(u(i))];
          [trueTangentSpace{i}, ~] = qr([cos(u(i)).*cos(v(i)), cos(u(i)).*sin(v(i)), -sin(u(i)); ...
            -sin(u(i)).*sin(v(i)), sin(u(i)).*cos(v(i)), 0]', 0);
      end
    case 'cone'
      sample = zeros(pointsNumber,3);
      k = 1;
      M = 4 / 3;
      f=@(fi,h)(h * (2 / 3 / pi));
      u = [];
      v = [];
      while k <= pointsNumber
          x0 = [rand(1)*pi, rand(1)+1];
          if rand(1) * M / pi <= f(x0(1), x0(2))
              u = [u; x0(1)];
              v = [v; x0(2)];
              sample(k,1) = cos(x0(1)) * x0(2);
              sample(k,2) = sin(x0(1)) * x0(2);
              sample(k,3)=x0(2);
              sample(k, :) = sample(k, :);
              [trueTangentSpace{k}, ~] = qr([-sin(x0(1)) * x0(2), cos(x0(1)) * x0(2), 0; cos(x0(1)), sin(x0(1)), 1]', 0);
              trueTangentSpace{k} = trueTangentSpace{k}; % * randMatrix; %
              k = k+1;
          end
      end
  end
  parametrization = [u, v];
end

function r=func_p(u1,v1)
  r=1/11.9045*sqrt( (cos(u1).^2.*cos(v1).^2+cos(u1).^2+1).*(sin(u1).^2.*sin(v1).^2+2.*sin(u1).^2) -(cos(u1).*sin(u1).*cos(v1).*sin(v1)).^2 );
end
