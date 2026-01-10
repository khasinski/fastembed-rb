# frozen_string_literal: true

module Fastembed
  # Async support for embedding operations
  #
  # Provides Future-like objects for running embeddings in background threads.
  # Useful for parallelizing embedding generation across multiple documents.
  #
  # @example Async embedding
  #   embedding = Fastembed::TextEmbedding.new
  #   future = embedding.embed_async(documents)
  #   # ... do other work ...
  #   vectors = future.value  # blocks until complete
  #
  # @example Multiple concurrent embeddings
  #   futures = documents.each_slice(100).map do |batch|
  #     embedding.embed_async(batch)
  #   end
  #   results = futures.flat_map(&:value)
  #
  module Async
    # A Future representing an async embedding operation
    #
    # Wraps a background thread that performs embedding, providing
    # methods to check completion status and retrieve results.
    #
    class Future
      # @return [Thread] The background thread
      attr_reader :thread

      # Create a new Future
      #
      # @yield Block to execute in background thread
      # @return [Future]
      def initialize(&block)
        @result = nil
        @error = nil
        @completed = false
        @mutex = Mutex.new
        @condition = ConditionVariable.new

        @thread = Thread.new do
          begin
            result = block.call
            @mutex.synchronize do
              @result = result
              @completed = true
              @condition.broadcast
            end
          rescue StandardError => e
            @mutex.synchronize do
              @error = e
              @completed = true
              @condition.broadcast
            end
          end
        end
      end

      # Check if the operation is complete
      #
      # @return [Boolean] True if complete (success or failure)
      def complete?
        @mutex.synchronize { @completed }
      end

      alias completed? complete?

      # Check if the operation is still running
      #
      # @return [Boolean] True if still running
      def pending?
        !complete?
      end

      # Check if the operation completed successfully
      #
      # @return [Boolean] True if completed without error
      def success?
        @mutex.synchronize { @completed && @error.nil? }
      end

      # Check if the operation failed
      #
      # @return [Boolean] True if completed with error
      def failure?
        @mutex.synchronize { @completed && !@error.nil? }
      end

      # Get the result, blocking until complete
      #
      # @param timeout [Numeric, nil] Maximum seconds to wait (nil = forever)
      # @return [Object] The result of the async operation
      # @raise [StandardError] If the operation raised an error
      # @raise [Timeout::Error] If timeout expires before completion
      def value(timeout: nil)
        wait(timeout: timeout)
        raise @error if @error

        @result
      end

      alias result value

      # Wait for completion without retrieving the result
      #
      # @param timeout [Numeric, nil] Maximum seconds to wait (nil = forever)
      # @return [Boolean] True if completed, false if timed out
      def wait(timeout: nil)
        @mutex.synchronize do
          return true if @completed

          if timeout
            deadline = Time.now + timeout
            until @completed
              remaining = deadline - Time.now
              break if remaining <= 0

              @condition.wait(@mutex, remaining)
            end
          else
            @condition.wait(@mutex) until @completed
          end

          @completed
        end
      end

      # Get the error if the operation failed
      #
      # @return [StandardError, nil] The error, or nil if successful/pending
      def error
        @mutex.synchronize { @error }
      end

      # Apply a transformation to the result
      #
      # @yield [result] Block to transform the result
      # @return [Future] A new Future with the transformed result
      def then(&block)
        Future.new do
          block.call(value)
        end
      end

      # Handle errors
      #
      # @yield [error] Block to handle errors
      # @return [Future] A new Future that handles errors
      def rescue(&block)
        Future.new do
          begin
            value
          rescue StandardError => e
            block.call(e)
          end
        end
      end
    end

    # Run multiple futures concurrently and wait for all to complete
    #
    # @param futures [Array<Future>] Futures to wait for
    # @param timeout [Numeric, nil] Maximum seconds to wait
    # @return [Array] Results from all futures
    # @raise [StandardError] If any future raised an error
    def self.all(futures, timeout: nil)
      futures.each { |f| f.wait(timeout: timeout) }
      futures.map(&:value)
    end

    # Run multiple futures concurrently and return first completed
    #
    # @param futures [Array<Future>] Futures to race
    # @param timeout [Numeric, nil] Maximum seconds to wait
    # @return [Object] Result from first completed future
    def self.race(futures, timeout: nil)
      deadline = timeout ? Time.now + timeout : nil

      loop do
        futures.each do |future|
          return future.value if future.complete?
        end

        if deadline && Time.now >= deadline
          raise Timeout::Error, 'No future completed within timeout'
        end

        sleep 0.001 # Small sleep to avoid busy waiting
      end
    end
  end
end
