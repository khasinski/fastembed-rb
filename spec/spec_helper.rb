# frozen_string_literal: true

require 'bundler/setup'
require 'fastembed'
require 'webmock/rspec'

RSpec.configure do |config|
  # Enable flags like --only-failures and --next-failure
  config.example_status_persistence_file_path = '.rspec_status'

  # Disable RSpec exposing methods globally on `Module` and `main`
  config.disable_monkey_patching!

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  # Allow real HTTP connections for integration tests
  config.before(:all, :integration) do
    WebMock.allow_net_connect!
  end

  config.after(:all, :integration) do
    WebMock.disable_net_connect!
  end

  config.before(:each, :integration) do
    WebMock.allow_net_connect!
  end

  config.after(:each, :integration) do
    WebMock.disable_net_connect!
  end
end
