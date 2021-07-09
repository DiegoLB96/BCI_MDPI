function varargout = rbmDefinitions(TypeLayer, TypeVisible)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   David Balderas 
%   03.12.2010
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RBM DEFINITIONS
%
% SYNTAX
% [f] = rbmDefinitions(TypeLayer)  creates structure with 
% different unit types
% 
% DEFINITION
% Creats a struc with the functions used in order to sample from 
% one layer to the other
%
% Abbreviations:
% I: Input of a unit
% R: Response of a unit (e.g. the activation probability)
% S: Sampling procedures of a unit
%
% The usual binary units work as follows:
% Input -> Response (logsig of input) -> Sample (compare response to
% uniform sample
% 
% 
% TypeLayer: could be Binary, Rlin, Gauss, noState
% %%% DESCRIPTION %%%
% Binary:     Binary units
% Rlin:       Rectified linear units
% Gauss:      Real Gaussian units
% Softmax:    Softmax units
% noState:    No sampling at all (simply return the response of the unit)
% Tansig:     Tangent sigmoid unit

if nargin < 1, TypeLayer = 'Binary'; end
if nargin < 2, TypeVisible = 'Binary'; end 
if ~ischar(TypeLayer) ||~ischar(TypeVisible), error('Types most be strings'); end
if nargout < 1, error('Must produce at least one structure as an output'); end

% State of Neurons
BinaryState    = @(R,I) R > rand(size(R));
RlinState      = @(R,I) max(0,I+randn(size(R)).*sqrt(logsig(I)));
GaussState     = @(R,I) I + randn(size(I));% linear unit with gaussian noise 
noState        = @(R,I) R;
SoftmaxState   = @(R,I) repmat(max(R,[],2),1,size(R,2)) == R;% cumsum(cumsum(R)>repmat(rand(size(R,2),1),size(R,1)),1) == 1; % 
        % TODO: SoftmaxState maybe wrong but one has to be activated
TansigState    = @(R,I) sign(R).*((abs(R)>rand(size(R)))*2-ones(size(R)));

% Activation or Reaction functions
BinaryReact     = @(I,S) ( 1./(1+ exp(-I )) ); % logistic Sigmoid
GaussReact      = @(I,S) I;     % Reaction = Input (linear unit)
                    % ( 1./(sqrt(2*pi) .* S).* exp(-1./(2*S.^2).* (I).^2 ));
noReact         = @(I,S) exp(-x.*x);
RlinReact       = @(I,S) log(1+exp(I))-log(1+exp(I-size(I,2))); % log(1+exp(I))-log(1+exp(I-nrect));
SoftmaxReact    = @(I,S) exp(I)./repmat(sum(exp(I),2),1,size(I,2));
TansigReact     = @(I,S) 2./(1+exp(-2*I))-1; % a little faster than tanh(I);


% only valid when there are more than one classes and its needed for a classification problem
hidInput = @(W,b,V,C,U) V*W + repmat(b,size(V,1),1) + C*U'; % Uses the data from previous and classes
visInput = @(W,a,H) H*W + repmat(a,size(H,1),1); % Just the reaction from hidden layer

%%%% Hidden Layer Functions %%%%%%
switch TypeLayer
    case 'Binary'
        State = BinaryState;
        React = BinaryReact;
    case 'Rlin'
        State = RlinState;
        React = RlinReact;
    case 'Gauss'
        State = GaussState;
        React = GaussReact;
    case 'No'
        State = noState;
        React = noReact;
    case 'Softmax'
        State = SoftmaxState;
        React = SoftmaxReact;
    case 'Tansig'
        State = TansigState;
        React = TansigReact;
    otherwise
        error('Not a Visible type allowed');
end
hidden =  struct('State', State, 'React', React, 'Input', hidInput);

%%%% Visible Layer Functions %%%%%%
switch TypeVisible
    case 'Binary'
        State = BinaryState;
        React = BinaryReact;
    case 'Rlin'
        State = RlinState;
        React = RlinReact;
    case 'Gauss'
        State = GaussState;
        React = GaussReact;
    case 'No'
        State = noState;
        React = noReact;
    case 'Softmax'
        State = SoftmaxState;
        React = SoftmaxReact;
    case 'Tansig'
        State = TansigState;
        React = TansigReact;
    otherwise
        error('Not a Visible type allowed');
end
visible = struct('State', State, 'React', React, 'Input', visInput);

varargout = {hidden, visible};
varargout = varargout(1:nargout);
    
end    
