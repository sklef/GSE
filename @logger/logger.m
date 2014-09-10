classdef logger < hgsetget
  %% logger class
  % Usage:
  %
  %   log = logger('Module name'); % default output to console, logLevel = fatal
  %   log = logger('Module name', logLevel.Info); % output to console, logLevel = info
  %   log = logger('Module name', logLevel.Info, 'log.txt'); % additionaly write to file
  %
  %   log.debug('Debug')
  %   log.info('Information')
  %   log.warn('Warning')
  %   log.error('Error')
  %   log.fatal('Fatal')
  
  properties
    outHandle
    moduleName = ''
    level = logLevel.Fatal % default 'Fatal' errors only
    isClosed
    print
  end
  
  properties (Constant)
    timeFormat = 13;
  end
  
  methods
    
    function log = logger(module, level, file)
      if nargin == 0
        error('Provide name of the logger')
      end
      
      if (nargin > 1)
        log.level = level;
      end
      
      if (nargin > 2)
        log.outHandle = fopen(file, 'a');
        log.print = @(type, str, varargin)log.printToAll(type, str, varargin{:});
      else
        log.outHandle = 1;
        log.print = @(type, str, varargin)logger.cprintf(type, str, varargin{:});
      end
      log.moduleName = strcat('\t{', module, '} ');
      log.isClosed = false;
    end
    
    function printToAll(log, type, str, varargin)
      fprintf(log.outHandle, str, varargin{:});
      logger.cprintf(type, str, varargin{:});
    end
    
    function debug(this, string, varargin)
      this.checkClosed();
      if this.level <= logLevel.Debug
        this.printTime()
        this.print('Comment', strcat(this.moduleName, ' [Debug]'))
        this.print('Text', [' ', string], varargin{:})
        this.print('Text', '\n')
      end
    end
    
    function info(this, string, varargin)
      this.checkClosed();
      if this.level <= logLevel.Info
        this.printTime()
        this.print('Keywords', strcat(this.moduleName, ' [Info]'))
        this.print('Text', [' ', string], varargin{:})
        this.print('Text', '\n')
      end
    end
    
    function warn(this, string, varargin)
      this.checkClosed();
      if this.level <= logLevel.Warn
        this.printTime()
        this.print('SystemCommands', strcat(this.moduleName, ' [Warn]'))
        this.print('Text', [' ', string], varargin{:})
        this.print('Text', '\n')
      end
    end
    
    function error(this, string, varargin)
      this.checkClosed();
      if this.level <= logLevel.Error
        this.printTime()
        this.print('Errors', strcat(this.moduleName, ' [Error]'))
        this.print('Text', [' ', string], varargin{:})
        this.print('Text', '\n')
      end
    end
    
    function fatal(this, string, varargin)
      this.checkClosed();
      this.printTime()
      this.print('Errors', strcat(this.moduleName, ' [Fatal]'))
      this.print('Text', [' ', string], varargin{:})
      this.print('Text', '\n')
    end
    
    function close(this)
      this.checkClosed();
      if this.outHandle ~= 1
        fclose(this.outHandle);
      end
      this.isClosed = true;
    end
    
    function checkClosed(this)
      if this.isClosed, 
        error('Logger is already closed!'); 
      end
    end
    
    function printTime(this)
      persistent lastChangeTime
      nowTime = datestr(now, logger.timeFormat);
      if ~strcmp(lastChangeTime, nowTime)
        this.print('Keywords', strcat('-> [', nowTime, ']\n'))
        lastChangeTime = nowTime;
      end
    end
    
  end
  
  methods (Static)
    
    count = cprintf(style,format,varargin)
    
  end
  
end